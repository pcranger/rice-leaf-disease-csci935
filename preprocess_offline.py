#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline preprocessing for Dhan-Shomadhan (Improved Version):
- Advanced HSV/LAB mask to keep leaf (green + yellow lesions + brown/tan lesions + white/gray scald)
- Apply CLAHE only INSIDE the leaf mask (optional)
- Optional resize
- Save to new root, preserving background & class folders

This version uses improved masking ranges based on practical rice leaf disease analysis.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# =============================================================================
# IMPORTANT VARIABLES - MODIFY THESE TO CHANGE PREPROCESSING CONFIGURATION
# =============================================================================
CLASS_NAMES = ['Brown Spot','Leaf Scald','Rice Blast','Rice Tungro','Sheath Blight']
CLASS_FIX = {
    'Leaf Scaled': 'Leaf Scald',
    'Brown spot': 'Brown Spot',
    'Sheath blight': 'Sheath Blight',
    'Rice tungro': 'Rice Tungro',
    'Rice blast': 'Rice Blast'
}
BG_DIRS = ['White Background', 'Field Background']

# -------------------- Args --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', type=str, default='./Dhan-Shomadhan', help='Original dataset root (Dhan-Shomadhan)')
    ap.add_argument('--out_dir', type=str, help='Processed dataset root (will be created under in_dir if not specified)')

    ap.add_argument('--mask', action='store_true', help='Apply background removal')
    ap.add_argument('--mask_mode', choices=['robust','green'], default='robust',
                    help='robust: keep green ∪ yellow + LAB; green: only green HSV range')
    ap.add_argument('--bg', choices=['white','black'], default='white',
                    help='Background color for removed area')
    ap.add_argument('--min_leaf_area_ratio', type=float, default=0.03,
                    help='If mask area < ratio of image area, skip masking (failsafe).')

    ap.add_argument('--clahe', action='store_true', help='Apply CLAHE (only inside mask if mask is used)')
    ap.add_argument('--resize', type=int, default=0, help='Resize to square (e.g., 224/320/448). 0=keep size')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1), help='Thread workers')

    ap.add_argument('--make_splits', action='store_true', help='Create stratified train/val/test under <out>_splits')
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--copy_mode', choices=['copy','move','link'], default='copy',
                    help='How to materialize split files.')

    # ✅ new: preview
    ap.add_argument('--preview', type=int, nargs='?', const=5,
                    help='Show N preview samples (no output written).')
    ap.add_argument('--crop_roi', action='store_true',
                help='Crop tight ROI around the largest leaf mask before resize.')
    ap.add_argument('--roi_pad', type=float, default=0.05,
                help='Padding ratio (0.0–0.3) added around the leaf bbox.')

    return ap.parse_args()

# -------------------- Utils --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path) -> List[Tuple[str, str, str]]:
    """
    Return list of (bg, cls, path) for all images.
    Works with either:
      - root/
          White Background/<Class>/*.jpg
          Field Background/<Class>/*.jpg
      - or a single background folder:
          Field Background/<Class>/*.jpg
    """
    items: List[Tuple[str, str, str]] = []

    def scan_bg_dir(bg_name: str, bg_dir: Path):
        if not bg_dir.exists():
            return
        for cls in os.listdir(bg_dir):
            cls_fixed = CLASS_FIX.get(cls, cls)
            if cls_fixed not in CLASS_NAMES:
                continue
            cdir = bg_dir / cls
            if not cdir.is_dir():
                continue
            for fn in os.listdir(cdir):
                if fn.startswith('.'):  # skip hidden
                    continue
                p = cdir / fn
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                    items.append((bg_name, cls_fixed, str(p)))

    # Case A: root has both background folders
    has_white = (root / 'White Background').exists()
    has_field = (root / 'Field Background').exists()

    if has_white or has_field:
        if has_white:
            scan_bg_dir('White Background', root / 'White Background')
        if has_field:
            scan_bg_dir('Field Background', root / 'Field Background')
        return items

    # Case B: root itself is a single background folder
    root_name = root.name
    if root_name in ('White Background', 'Field Background'):
        scan_bg_dir(root_name, root)
        return items

    # Fallback: nothing matched
    print(f"[WARN] '{root}' does not look like a dataset root or a background folder.")
    return items

# -------------------- Improved Masking --------------------

def mask_green_only(rgb: np.ndarray) -> np.ndarray:
    """Simple green mask for basic cases."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    g1 = np.array([25, 30, 30]); g2 = np.array([95, 255, 255])
    return cv2.inRange(hsv, g1, g2)

def get_leaf_roi(rgb: np.ndarray, is_white_bg: bool = False) -> np.ndarray:
    """
    Step 1: Get coarse foreground (leaf) ROI
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    
    if is_white_bg:
        # White background images: foreground = S > 10 OR V < 240
        foreground = (S > 10) | (V < 240)
        foreground = foreground.astype(np.uint8) * 255
        
        # Keep largest contour
        contours_info = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            leaf_roi = np.zeros_like(foreground)
            cv2.drawContours(leaf_roi, [biggest], -1, 255, thickness=cv2.FILLED)
            return leaf_roi
        return foreground
    else:
        # Field images: green leaf pre-mask
        green_mask = (H >= 30) & (H <= 85) & (S > 40) & (V > 40)
        green_mask = green_mask.astype(np.uint8) * 255
        
        # Morphological operations
        kernel = np.ones((5, 7), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep connected components with elongated shape (leaf blades)
        contours_info = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        
        if cnts:
            # Filter by aspect ratio (leaf blades are elongated)
            valid_contours = []
            for cnt in cnts:
                if cv2.contourArea(cnt) > 100:  # Minimum area
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = max(w, h) / max(min(w, h), 1)
                    if aspect_ratio > 1.5:  # Elongated shape
                        valid_contours.append(cnt)
            
            if valid_contours:
                # Combine valid contours
                leaf_roi = np.zeros_like(green_mask)
                cv2.drawContours(leaf_roi, valid_contours, -1, 255, thickness=cv2.FILLED)
                
                # Add LAB-a support for dull greens
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                a = lab[:, :, 1]
                lab_mask = (a < 125).astype(np.uint8) * 255
                leaf_roi = cv2.bitwise_or(leaf_roi, lab_mask)
                
                return leaf_roi
        
        return green_mask

def get_disease_masks(rgb: np.ndarray, leaf_roi: np.ndarray) -> np.ndarray:
    """
    Step 2: Get disease masks (within leaf ROI only)
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    
    # A. Leaf tissue (healthy green to light chlorosis)
    leaf_green = (H >= 35) & (H <= 82) & (S >= 60) & (V >= 60)
    leaf_yellow = (H >= 18) & (H <= 45) & (S >= 60) & (V >= 90)  # Tungro yellowing
    leaf_mask = (leaf_green | leaf_yellow).astype(np.uint8) * 255
    
    # B. Brown/tan lesions (two windows for reds/browns)
    brown1 = (H >= 0) & (H <= 10) & (S >= 50) & (V >= 50) & (V <= 220)
    brown2 = (H >= 10) & (H <= 30) & (S >= 50) & (V >= 50) & (V <= 220)
    brown_mask = (brown1 | brown2).astype(np.uint8) * 255
    
    # C. White/gray scald (apply only inside leaf ROI)
    white_lesion = ((S <= 40) & (V >= 160)).astype(np.uint8) * 255
    
    # Restrict to leaf ROI
    lesions = cv2.bitwise_and(cv2.bitwise_or(brown_mask, white_lesion), leaf_roi)
    
    # Final mask = leaf + lesions
    final_mask = cv2.bitwise_or(leaf_mask, lesions)
    
    # Additional cleanup: ensure lesions are connected to leaf tissue
    # Dilate leaf ROI slightly to include nearby lesions
    kernel = np.ones((5, 5), np.uint8)
    dilated_roi = cv2.dilate(leaf_roi, kernel, iterations=1)
    lesions = cv2.bitwise_and(lesions, dilated_roi)
    
    final_mask = cv2.bitwise_or(leaf_mask, lesions)
    
    return final_mask

def mask_robust_leaf_keep_lesions_v2(rgb: np.ndarray) -> np.ndarray:
    """
    Improved robust masking using the new approach.
    """
    # Determine if this is a white background image
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    is_white_bg = np.mean(V) > 200 and np.mean(S) < 50
    
    # Step 1: Get leaf ROI
    leaf_roi = get_leaf_roi(rgb, is_white_bg)
    
    # Step 2: Get disease masks within leaf ROI
    disease_mask = get_disease_masks(rgb, leaf_roi)
    
    # Step 3: Union & cleanup
    final_mask = cv2.bitwise_or(leaf_roi, disease_mask)
    
    # Morphology: open(3) then close(5)
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill holes on long, thin components
    contours_info = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    
    if cnts:
        filled_mask = np.zeros_like(final_mask)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 50:  # Minimum area
                # Check if it's a long, thin component (sheath-blight/leaf-scald streaks)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio > 6 and min(w, h) >= 3 and min(w, h) <= 20:
                    # Fill holes for long, thin components
                    cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                else:
                    cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        final_mask = filled_mask
    
    # Keep largest component as failsafe
    contours_info = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        mask2 = np.zeros_like(final_mask)
        cv2.drawContours(mask2, [biggest], -1, 255, thickness=cv2.FILLED)
        final_mask = mask2
    
    return final_mask

def build_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'robust':
        return mask_robust_leaf_keep_lesions_v2(rgb)
    else:
        return mask_green_only(rgb)

# -------------------- CLAHE (inside mask) --------------------

def apply_clahe_rgb(rgb: np.ndarray, clip=2.0, tiles=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    return out

# -------------------- One-file processing --------------------

def preprocess_one(in_path: str, out_path: str, args) -> Tuple[bool, str]:
    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return False, f"read-fail:{in_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    H, W = rgb.shape[:2]
    bg_val = 255 if args.bg == 'white' else 0

    out_rgb = rgb.copy()

    if args.mask:
        mask = build_mask(rgb, args.mask_mode)
        area = int((mask > 0).sum())
        if area < args.min_leaf_area_ratio * (H * W):
            mask = None
        if mask is not None:
            # Apply mask: keep leaf + lesions, replace background
            out_rgb = rgb.copy()
            out_rgb[mask == 0] = [bg_val, bg_val, bg_val]
            
            # Apply CLAHE only inside mask if requested
            if args.clahe:
                clahe_rgb = apply_clahe_rgb(rgb)
                out_rgb[mask > 0] = clahe_rgb[mask > 0]
            
            # ====== NEW: crop ROI quanh bbox lớn nhất của mask ======
            if getattr(args, 'crop_roi', False):
                ys, xs = np.where(mask > 0)
                if ys.size > 0 and xs.size > 0:
                    y0, y1 = ys.min(), ys.max() + 1
                    x0, x1 = xs.min(), xs.max() + 1
                    # pad theo tỉ lệ cạnh dài
                    h, w = y1 - y0, x1 - x0
                    pad = int(max(h, w) * float(getattr(args, 'roi_pad', 0.05)))
                    y0 = max(0, y0 - pad); y1 = min(mask.shape[0], y1 + pad)
                    x0 = max(0, x0 - pad); x1 = min(mask.shape[1], x1 + pad)
                    # crop cả ảnh và mask
                    out_rgb = out_rgb[y0:y1, x0:x1]
        elif args.clahe:
            out_rgb = apply_clahe_rgb(rgb)
    elif args.clahe:
        out_rgb = apply_clahe_rgb(rgb)

    if args.resize and args.resize > 0:
        out_rgb = cv2.resize(out_rgb, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(out_path, out_bgr)
    if not ok:
        return False, f"write-fail:{out_path}"
    return True, out_path

# -------------------- Batch processing --------------------

def preprocess_all(in_root: Path, out_root: Path, args):
    items = list_images(in_root)
    print(f"[INFO] Found {len(items)} images.")
    ensure_dir(out_root)

    def out_path_for(bg, cls, in_path):
        rel = os.path.basename(in_path)
        dst_dir = out_root / bg / cls
        ensure_dir(dst_dir)
        return str(dst_dir / rel)

    ok_count, fail_count = 0, 0
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for bg, cls, p in items:
            dst = out_path_for(bg, cls, p)
            futures.append(ex.submit(preprocess_one, p, dst, args))
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            ok, info = f.result()
            if ok: ok_count += 1
            else:
                fail_count += 1
                print(f"[ERR] {info}")
    print(f"[DONE] Preprocessed OK={ok_count}, FAIL={fail_count}")
    return ok_count, fail_count

# -------------------- Splits --------------------

def stratified_splits(pre_root: Path, out_root: Path, val_ratio: float, test_ratio: float, seed: int, copy_mode: str):
    import numpy as np
    rng = np.random.RandomState(seed)
    ensure_dir(out_root)

    items = list_images(pre_root)
    if len(items) == 0:
        print("[WARN] No preprocessed images to split.")
        return

    by_bg: Dict[str, List[Tuple[str,str,str]]] = {'White Background': [], 'Field Background': []}
    for bg, cls, p in items:
        if bg in by_bg:
            by_bg[bg].append((bg, cls, p))

    def split_one_domain(domain_items: List[Tuple[str, str, str]]):
        by_cls: Dict[str, List[Tuple[str, str, str]]] = {c: [] for c in CLASS_NAMES}
        for bg, cls, p in domain_items:
            by_cls[cls].append((bg, cls, p))
        train, val, test = [], [], []
        for cls in CLASS_NAMES:
            arr = by_cls[cls]
            if not arr:
                continue
            idx = np.arange(len(arr))
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(round(n * test_ratio))  # 10% for testing
            n_val = int(round(n * val_ratio))  # 20% for validation
            n_train = n - n_test - n_val  # Remaining 70% for training
            test_idx = idx[:n_test]
            val_idx = idx[n_test:n_test + n_val]
            train_idx = idx[n_test + n_val:]
            train += [arr[i] for i in train_idx]
            val += [arr[i] for i in val_idx]
            test += [arr[i] for i in test_idx]
        return train, val, test

    def materialize(split, split_name: str, bg_filter: str=None):
        for bg, cls, src in split:
            bg_out = bg if bg_filter is None else bg_filter
            dst = out_root / split_name / bg_out / cls / os.path.basename(src)
            ensure_dir(dst.parent)
            if copy_mode == 'copy':
                shutil.copy2(src, dst)
            elif copy_mode == 'move':
                shutil.move(src, dst)
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

    for bg in BG_DIRS:
        if len(by_bg[bg]) == 0:
            print(f"[WARN] No images for {bg}. Skipping split.")
            continue
        tr, va, te = split_one_domain(by_bg[bg])
        print(f"[SPLIT] {bg}: train={len(tr)}, val={len(va)}, test={len(te)}")
        materialize(tr, 'train', bg_filter=bg)
        materialize(va, 'val',   bg_filter=bg)
        materialize(te, 'test',  bg_filter=bg)

    for split_name in ['train','val','test']:
        lst = []
        split_dir = out_root / split_name
        if split_dir.exists():
            for bg in BG_DIRS:
                d = split_dir / bg
                if not d.exists(): continue
                for cls in CLASS_NAMES:
                    cdir = d / cls
                    if not cdir.exists(): continue
                    for fn in os.listdir(cdir):
                        lst.append(str(cdir / fn))
        list_path = out_root / f'list_{split_name}.txt'
        with open(list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lst))
        print(f"[LIST] Wrote {list_path} ({len(lst)} files)")

# -------------------- Preview --------------------

def preview_samples(in_root: Path, args, num_preview: int = 5):
    import random, matplotlib.pyplot as plt
    from PIL import Image
    import time

    items = list_images(in_root)
    if len(items) == 0:
        print("[preview] No images found.")
        return
    
    # Use current time as seed for true randomness each time
    random.seed(int(time.time() * 1000) % 2**32)
    samples = random.sample(items, min(num_preview, len(items)))
    print(f"[preview] Showing {len(samples)} random samples with current settings...")

    for i, (bg, cls, path) in enumerate(samples, 1):
        rgb = np.array(Image.open(path).convert('RGB'))
        tmp_ok, _ = preprocess_one(path, 'tmp_preview.jpg', args)
        proc = cv2.imread('tmp_preview.jpg') if tmp_ok else rgb[..., ::-1]
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10,4))
        plt.suptitle(f"[{bg}] {cls}", fontsize=12)
        plt.subplot(1,2,1); plt.imshow(rgb); plt.title("Original"); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(proc); plt.title("Preprocessed"); plt.axis('off')
        plt.tight_layout(); plt.show()

# -------------------- Main --------------------

def main():
    args = parse_args()
    in_root = Path(args.in_dir)
    
    # If out_dir is not specified, create it under the main dataset folder
    if args.out_dir is None:
        out_root = in_root / "preprocessed"
    else:
        out_root = Path(args.out_dir)
        # Ensure out_dir is under the main dataset folder
        if not str(out_root).startswith(str(in_root)):
            out_root = in_root / args.out_dir
    
    ensure_dir(out_root)

    if args.preview is not None:
        preview_samples(in_root, args, num_preview=args.preview)
        print("[preview] Done. To run full preprocessing, remove --preview.")
        return

    print(f"[CFG] in={in_root} out={out_root} mask={args.mask} mode={args.mask_mode} "
          f"bg={args.bg} clahe={args.clahe} resize={args.resize} workers={args.workers} "
          f"min_leaf_area_ratio={args.min_leaf_area_ratio}")

    preprocess_all(in_root, out_root, args)

    if args.make_splits:
        split_root = in_root / "preprocessed_splits"
        print(f"[SPLIT] Creating splits at: {split_root}")
        stratified_splits(out_root, split_root, args.val_ratio, args.test_ratio, args.seed, args.copy_mode)

    print("[DONE] offline preprocessing completed.")

if __name__ == "__main__":
    main()

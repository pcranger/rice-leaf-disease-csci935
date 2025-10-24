#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline preprocessing for Dhan-Shomadhan:
- Robust HSV/LAB mask to keep leaf (green + yellow lesions), with morphology & largest CC
- Apply CLAHE only INSIDE the leaf mask (optional)
- Optional resize
- Save to new root, preserving background & class folders

Optional: create stratified splits with fixed seed.
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
    ap.add_argument('--in_dir', type=str, required=True, help='Original dataset root (Dhan-Shomadhan)')
    ap.add_argument('--out_dir', type=str, required=True, help='Processed dataset root')

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


# -------------------- Masking --------------------

def mask_green_only(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    g1 = np.array([25, 30, 30]); g2 = np.array([95, 255, 255])
    return cv2.inRange(hsv, g1, g2)

def mask_robust_leaf_keep_lesions(rgb: np.ndarray) -> np.ndarray:
    """Keep green + yellow lesions + LAB-a support; fix holes & keep largest CC."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # green (wider) + yellow to keep lesions
    # g1 = np.array([20, 25, 25]); g2 = np.array([100, 255, 255])
    # y1 = np.array([8,  20, 25]); y2 = np.array([40,  255, 255])
    # trong mask_robust_leaf_keep_lesions
    g1 = np.array([20, 60, 25]); g2 = np.array([100, 255, 255])  
    y1 = np.array([10, 60, 25]); y2 = np.array([40,  255, 255])  

    mask_g = cv2.inRange(hsv, g1, g2)
    mask_y = cv2.inRange(hsv, y1, y2)
    mask = cv2.bitwise_or(mask_g, mask_y)

    # LAB-a low (greenish)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1]
    # mask_lab = cv2.inRange(a, 0, 150)
    mask_lab = cv2.inRange(a, 0, 135)
    mask = cv2.bitwise_or(mask, mask_lab)

    # morphology: close -> open
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    # keep largest component
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        mask2 = np.zeros_like(mask)
        cv2.drawContours(mask2, [biggest], -1, 255, thickness=cv2.FILLED)
        mask = mask2
    return mask

def build_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    return mask_robust_leaf_keep_lesions(rgb) if mode == 'robust' else mask_green_only(rgb)

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
                    rgb_crop  = rgb[y0:y1, x0:x1]
                    mask_crop = mask[y0:y1, x0:x1]
                else:
                    rgb_crop, mask_crop = rgb, mask
            else:
                rgb_crop, mask_crop = rgb, mask
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

    def split_one_domain(domain_items: List[Tuple[str,str,str]]):
        by_cls: Dict[str, List[Tuple[str,str,str]]] = {c: [] for c in CLASS_NAMES}
        for bg, cls, p in domain_items:
            by_cls[cls].append((bg, cls, p))
        train, val, test = [], [], []
        for cls in CLASS_NAMES:
            arr = by_cls[cls]
            if not arr: continue
            idx = np.arange(len(arr))
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(round(n * test_ratio))
            n_val  = int(round((n - n_test) * val_ratio))
            test_idx = idx[:n_test]
            val_idx  = idx[n_test:n_test+n_val]
            train_idx= idx[n_test+n_val:]
            train += [arr[i] for i in train_idx]
            val   += [arr[i] for i in val_idx]
            test  += [arr[i] for i in test_idx]
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

    items = list_images(in_root)
    if len(items) == 0:
        print("[preview] No images found.")
        return
    random.seed(args.seed)
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
    in_root  = Path(args.in_dir)
    out_root = Path(args.out_dir)
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
        split_root = Path(str(out_root) + "_splits")
        print(f"[SPLIT] Creating splits at: {split_root}")
        stratified_splits(out_root, split_root, args.val_ratio, args.test_ratio, args.seed, args.copy_mode)

    print("[DONE] offline preprocessing completed.")

if __name__ == "__main__":
    main()

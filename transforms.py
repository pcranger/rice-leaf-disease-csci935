# transforms.py — optimized
from typing import Tuple
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img: Image.Image):
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class HSVGreenMask:
    def __init__(self, lower=(25, 40, 40), upper=(85, 255, 255)):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
    def __call__(self, img: Image.Image):
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        fg = arr.copy()
        bg = np.full_like(arr, 255)
        out = np.where(mask[..., None] > 0, fg, bg)
        return Image.fromarray(out)


def build_transforms(img_size: int, use_clahe=True, use_green_mask=False,
                     color_jitter=True, blur=True, perspective=True,
                     heavy_eval=False) -> Tuple[T.Compose, T.Compose]:
    """
    Return train_tf, eval_tf.
    heavy_eval=False => skip expensive preprocess in eval to save time.
    """
    augs = []
    if use_clahe:
        augs.append(CLAHE())
    if use_green_mask:
        augs.append(HSVGreenMask())

    # --- train augmentations ---
    train_tf = [
        *augs,
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
    ]
    if perspective:
        train_tf.append(T.RandomPerspective(distortion_scale=0.25, p=0.15))  # reduced p
    if color_jitter:
        train_tf.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if blur:
        train_tf.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))

    train_tf += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # --- eval transform (simplified) ---
    eval_aug = []
    if heavy_eval:
        eval_aug = augs  # optional heavy preprocessing
    eval_tf = T.Compose([
        *eval_aug,
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return T.Compose(train_tf), eval_tf

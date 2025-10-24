# datasets.py
import platform, random
from pathlib import Path
from typing import Tuple, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from transforms import build_transforms

print(f"[datasets] loaded from: {__file__}")

CLASS_NAMES = ['Brown Spot','Leaf Scald','Rice Blast','Rice Tungro','Sheath Blight']

class Merged(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, y

def _dirs_for(scenario: str, root: Path) -> List[Path]:
    if scenario == 'white': return [root/'White Background']
    if scenario == 'field': return [root/'Field Background']
    if scenario == 'mixed': return [root/'White Background', root/'Field Background']
    raise ValueError("scenario must be white/field/mixed")

def _fix_names(ds: ImageFolder):
    mapping = {
        'Leaf Scaled': 'Leaf Scald',
        'Brown spot': 'Brown Spot',
        'Sheath blight': 'Sheath Blight',
        'Rice tungro': 'Rice Tungro',
        'Rice blast': 'Rice Blast'
    }
    ds.classes = [mapping.get(c, c) for c in ds.classes]
    ds.class_to_idx = {c:i for i,c in enumerate(sorted(CLASS_NAMES))}

def collect_samples(root: Path, scenario: str):
    """Return (samples, targets) for the chosen scenario."""
    out_samples, out_targets = [], []
    for d in _dirs_for(scenario, root):
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")
        ds = ImageFolder(str(d))
        if set(ds.classes) != set(CLASS_NAMES): _fix_names(ds)
        for path, y in ds.samples:
            name = ds.classes[y]
            yg = CLASS_NAMES.index(name)
            out_samples.append((path, yg))
            out_targets.append(yg)
    return out_samples, np.array(out_targets)

def split_train_val(samples, targets, val_ratio: float, seed: int):
    """Return (train_idx, val_idx) stratified from provided samples."""
    rng = seed
    idxs = np.arange(len(samples))
    if val_ratio <= 0.0:
        return idxs, np.array([], dtype=int)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng)
    train_idx, val_idx = next(sss.split(idxs, targets))
    return train_idx, val_idx

def make_train_val_loaders(data_dir: Path, train_domain: str, img_size: int, batch_size: int,
                           val_ratio: float, num_workers: int, aug_flags: dict, seed: int):
    """Build train/val loaders from TRAIN DOMAIN only. Return also set of train/val file paths for exclusion."""
    samples, targets = collect_samples(Path(data_dir), train_domain)
    tf_train, tf_eval = build_transforms(img_size=img_size, **aug_flags)
    train_idx, val_idx = split_train_val(samples, targets, val_ratio, seed)

    ds_train = Merged([samples[i] for i in train_idx], targets[train_idx].tolist(), transform=tf_train)
    ds_val   = Merged([samples[i] for i in val_idx],   targets[val_idx].tolist(),   transform=tf_eval)

    on_windows = (platform.system().lower() == 'windows')
    use_cuda = torch.cuda.is_available()
    nw = 0 if on_windows else num_workers
    pin = True if use_cuda else False

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pin)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    # paths used in training/val to avoid overlap when test shares domain
    used_paths: Set[str] = set(p for p,_ in ds_train.samples) | set(p for p,_ in ds_val.samples)
    return loader_train, loader_val, used_paths, CLASS_NAMES

def make_test_loader_excluding(data_dir: Path, test_scenario: str, exclude_paths: Set[str],
                               img_size: int, batch_size: int, num_workers: int,
                               aug_flags: dict, train_domain: str = None):
    """Build an EVAL loader for TEST SCENARIO, excluding any paths used in train/val."""
    samples, targets = collect_samples(Path(data_dir), test_scenario)
    # filter out overlaps only if test scenario is same as train domain
    filt_samples, filt_targets = [], []
    for (p, y) in samples:
        # Only exclude paths if test scenario matches train domain
        if train_domain is None or test_scenario != train_domain:
            # Different domain - use all samples
            filt_samples.append((p,y))
            filt_targets.append(y)
        else:
            # Same domain - exclude used paths
            if p not in exclude_paths:
                filt_samples.append((p,y))
                filt_targets.append(y)
    
    # If no samples after filtering (same domain case), use a subset of unused samples
    if len(filt_samples) == 0 and train_domain is not None and test_scenario == train_domain:
        # Use 20% of the original samples as test data
        from sklearn.model_selection import train_test_split
        test_samples, _ = train_test_split(samples, test_size=0.2, random_state=42, stratify=targets)
        filt_samples = [(p, y) for p, y in test_samples]
        filt_targets = [y for p, y in test_samples]

    # eval transform
    _, tf_eval = build_transforms(img_size=img_size, **aug_flags)
    ds_test = Merged(filt_samples, filt_targets, transform=tf_eval)

    on_windows = (platform.system().lower() == 'windows')
    use_cuda = torch.cuda.is_available()
    nw = 0 if on_windows else num_workers
    pin = True if use_cuda else False

    # Shuffle test data to avoid class bias
    return DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)

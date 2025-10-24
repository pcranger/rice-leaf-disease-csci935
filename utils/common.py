import torch, os
from pathlib import Path

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
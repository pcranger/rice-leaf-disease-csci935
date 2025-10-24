from dataclasses import dataclass
from pathlib import Path
import platform

@dataclass
class Config:
    # data_dir: Path = Path('./Dhan-Shomadhan')
    data_dir: Path = Path('./preprocessed') 
    out_dir: Path = Path('./out')

    # NEW: chọn domain dùng để TRAIN (test luôn chạy đủ 3 scenario)
    train_domain: str = 'mixed'   # 'white' | 'field' | 'mixed'

    # training
    epochs: int = 8
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2 if platform.system().lower()=='windows' else 4
    amp: bool = True

    # model + finetune
    model_name: str = 'resnet50'
    finetune_mode: str = 'full_ft'
    drop_rate: float = 0.2
    img_size: int = 224

    # eval repeats
    num_runs: int = 5
    seed0: int = 45

    # split ratios for TRAIN DOMAIN ONLY
    val_ratio: float = 0.2
    test_ratio: float = 0.0   # ← train-only split; test sẽ tạo RIÊNG theo 3 scenario

    # augs
    use_clahe: bool      = False              # nếu đã offline preprocess
    use_green_mask: bool = False
    blur: bool           = True
    perspective: bool    = True
    color_jitter: bool   = True


    

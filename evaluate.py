from pathlib import Path
import json
from typing import Dict

import torch
from utils.metrics import evaluate_on_loader
from utils.common import device


def evaluate_saved(model, loader_test, ckpt_path: Path) -> Dict[str, float]:
    dev = device()
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt['state_dict'])
    cm_test, report_test, macro_f1_test = evaluate_on_loader(model, loader_test, dev)
    out = {
        'test_f1': macro_f1_test,
        'classification_report': report_test,
        'confusion_matrix': cm_test.tolist()
    }
    with open(ckpt_path.with_suffix('.eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out
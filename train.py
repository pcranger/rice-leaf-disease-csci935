# train.py — optimized version
from pathlib import Path
from typing import Dict, Mapping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.common import device
from utils.metrics import evaluate_on_loader
from utils.plots import plot_curves


def validate_epoch(model, val_loader, device, criterion):
    """Single-pass validation: returns avg_loss, macro_f1."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, 1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.cpu().numpy().tolist())
    import numpy as np
    from sklearn.metrics import f1_score
    avg_loss = total_loss / max(1, len(val_loader.dataset))
    macro_f1 = f1_score(np.array(y_true), np.array(y_pred), average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_then_eval_multi(model, train_loader, val_loader,
                          test_loaders: Mapping[str, torch.utils.data.DataLoader],
                          cfg, run_name: str, out_dir: Path):
    """
    Train once (train/val), evaluate on multiple test scenarios.
    Returns: dict {scenario: test_f1}
    """
    dev = device()
    model = model.to(dev)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # ✅ new AMP API
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")

    best_f1 = -1.0
    best_path = out_dir / f"{run_name}_best.pt"
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and dev.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)

        avg_train = running / len(train_loader.dataset)
        avg_val, macro_f1 = validate_epoch(model, val_loader, dev, criterion)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1"].append(macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            out_dir.mkdir(parents=True, exist_ok=True)
            # ✅ Only compute report/CM once when saving best
            cm, report, _ = evaluate_on_loader(model, val_loader, dev, zero_division=0)
            torch.save({"state_dict": model.state_dict(), "report": report}, best_path)

        scheduler.step()

    # ✅ load best model
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # ✅ evaluate ONCE on all test scenarios
    results = {}
    for scen, test_loader in test_loaders.items():
        cm_t, rep_t, f1_t = evaluate_on_loader(model, test_loader, dev, zero_division=0)
        results[scen] = {"test_f1": f1_t, "report": rep_t, "cm": cm_t.tolist()}

        import json
        with open(out_dir / f"{run_name}_{scen}_test_report.json", "w") as f:
            json.dump(rep_t, f, indent=2)

    # ✅ save curves
    plot_curves(history, out_dir, title_prefix=run_name)
    return results

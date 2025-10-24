# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# debug: xác nhận đúng file được import
print(f"[metrics] loaded from: {__file__}")

def _class_hist(y, num_classes=5):
    y = np.asarray(y, dtype=np.int64)
    return np.bincount(y, minlength=num_classes)

def evaluate_on_loader(model, loader, device, num_classes=5, zero_division=0):
    """
    Evaluate model on a given loader.
    Returns: (confusion_matrix, classification_report_dict, macro_f1)

    - zero_division: 0 to avoid UndefinedMetricWarning and set metrics to 0 when a class has no preds/targets.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # in phân bố lớp để dễ debug thiếu lớp
    true_hist = _class_hist(y_true, num_classes=num_classes)
    pred_hist = _class_hist(y_pred, num_classes=num_classes)
    print(f"[metrics] true per-class: {true_hist.tolist()} | pred per-class: {pred_hist.tolist()}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=zero_division
    )
    macro_f1 = f1_score(
        y_true, y_pred,
        average='macro',
        zero_division=zero_division
    )
    return cm, report, macro_f1

from pathlib import Path
import matplotlib.pyplot as plt


def plot_curves(history, out_dir: Path, title_prefix='run'):
    out_dir.mkdir(parents=True, exist_ok=True)
    # history: dict with lists: train_loss, val_loss, val_f1
    for key in ['train_loss','val_loss','val_f1']:
        plt.figure()
        plt.plot(history[key])
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.title(f'{title_prefix} — {key}')
        plt.grid(True)
        plt.savefig(out_dir / f'{title_prefix}_{key}.png', dpi=150, bbox_inches='tight')
        plt.close()
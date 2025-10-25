# compare.py
from pathlib import Path
import statistics

from config import Config
from utils.common import ensure_dir, set_seed
from datasets import make_train_val_loaders, make_test_loader_excluding
from models import build_model
from train import train_then_eval_multi

TEST_SCENARIOS = ['white','field','mixed']
# MODELS = ['resnet50','efficientnetb0','mobilenetv2']
MODELS = ['resnet50']



def run_suite_cross(cfg: Config):
    ensure_dir(cfg.out_dir)
    rows = []  # (train_domain, model, scen, mean, std, per_run)

    for model_name in MODELS:
        per_scen_runs = {s: [] for s in TEST_SCENARIOS}

        for k in range(cfg.num_runs):
            seed = cfg.seed0 + k
            set_seed(seed)

            aug_flags = dict(
                use_clahe=cfg.use_clahe,
                use_green_mask=cfg.use_green_mask,
                color_jitter=cfg.color_jitter,
                blur=cfg.blur,
                perspective=cfg.perspective
            )

            # Build TRAIN/VAL from cfg.train_domain
            train_loader, val_loader, used_paths, class_names = make_train_val_loaders(
                cfg.data_dir, cfg.train_domain, cfg.img_size, cfg.batch_size,
                cfg.val_ratio, cfg.num_workers, aug_flags, seed
            )

            # Build 3 TEST loaders (exclude train/val paths to guarantee no leakage)
            test_loaders = {}
            for scen in TEST_SCENARIOS:
                tl = make_test_loader_excluding(cfg.data_dir, scen, used_paths,
                                                cfg.img_size, cfg.batch_size,
                                                cfg.num_workers, aug_flags, cfg.train_domain)
                test_loaders[scen] = tl

            # Train ONCE then evaluate on all tests
            net = build_model(model_name, cfg.finetune_mode, cfg.drop_rate)
            run_name = f"train-{cfg.train_domain}_{model_name}_seed{seed}"
            out_dir = Path(cfg.out_dir) / f"train-{cfg.train_domain}" / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            results = train_then_eval_multi(net, train_loader, val_loader, test_loaders, cfg, run_name, out_dir)

            for scen in TEST_SCENARIOS:
                per_scen_runs[scen].append(results[scen]['test_f1'])
                print(f"[{model_name}] seed={seed} | train={cfg.train_domain} | test={scen} | F1={results[scen]['test_f1']:.4f}")

        # aggregate per scenario
        for scen in TEST_SCENARIOS:
            mean_f1 = statistics.mean(per_scen_runs[scen])
            std_f1  = statistics.pstdev(per_scen_runs[scen])
            rows.append((cfg.train_domain, model_name, scen, mean_f1, std_f1, per_scen_runs[scen]))
            print(f"AGG | model={model_name} | train={cfg.train_domain} | test={scen} | F1={mean_f1:.4f} ± {std_f1:.4f} {per_scen_runs[scen]}")

    # Save table
    import csv
    table_path = Path(cfg.out_dir) / f'summary_train-{cfg.train_domain}.csv'
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TrainDomain','Model','TestScenario','Mean Test Macro-F1','Std','Per-run'])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], f"{row[3]:.4f}", f"{row[4]:.4f}", ' '.join(f"{x:.4f}" for x in row[5])])
    print(f"Saved: {table_path}")

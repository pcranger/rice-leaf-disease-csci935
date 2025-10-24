from config import Config
from compare import run_suite_cross

if __name__ == '__main__':
    cfg = Config()
    # Ví dụ: train chỉ white
    # cfg.train_domain = 'white'
    # Chạy nhanh
    # cfg.epochs = 5; cfg.num_runs = 2
    run_suite_cross(cfg)

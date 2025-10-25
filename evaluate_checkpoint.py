#!/usr/bin/env python3
"""
Batch checkpoint evaluation script.
Evaluates multiple models and scenarios automatically.
"""
from pathlib import Path
import torch
import csv
from datetime import datetime
from models import build_model
from utils.metrics import evaluate_on_loader
from utils.common import device
from datasets import make_test_loader_excluding
from config import Config
from PIL import Image
from torchvision import transforms
import argparse

# =============================================================================
# IMPORTANT VARIABLES - MODIFY THESE TO CHANGE EVALUATION SETUP
# =============================================================================
MODELS = ['efficientnetb0', 'mobilenetv2', 'resnet50']
TEST_SCENARIOS = ['white', 'field', 'mixed']
SEEDS = [42, 43, 44, 45, 46]  # Available seeds
TRAIN_DOMAIN = 'mixed'  # Training domain used


def evaluate_checkpoint(checkpoint_path: Path, test_scenario: str = 'mixed'):
    """Evaluate a saved checkpoint on a specific test scenario."""
    cfg = Config()
    
    # Extract model name from checkpoint path
    model_name = checkpoint_path.parent.name
    
    # Build model
    model = build_model(model_name, cfg.finetune_mode, cfg.drop_rate)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device())
    
    # Create test loader
    aug_flags = dict(
        use_clahe=cfg.use_clahe,
        use_green_mask=cfg.use_green_mask,
        color_jitter=cfg.color_jitter,
        blur=cfg.blur,
        perspective=cfg.perspective
    )
    
    test_loader = make_test_loader_excluding(
        cfg.data_dir, test_scenario, [],  # empty used_paths
        cfg.img_size, cfg.batch_size, cfg.num_workers,
        aug_flags, cfg.train_domain
    )
    
    # Evaluate
    cm, report, f1_score = evaluate_on_loader(model, test_loader, device())
    
    # Print results
    print(f"Model: {model_name}")
    print(f"Test Scenario: {test_scenario}")
    print(f"F1 Score: {f1_score:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save to CSV
    save_to_csv(checkpoint_path, model_name, test_scenario, f1_score, report)
    
    return {
        'model': model_name,
        'scenario': test_scenario,
        'f1_score': f1_score,
        'report': report,
        'confusion_matrix': cm
    }


def save_to_csv(checkpoint_path: Path, model_name: str, test_scenario: str, f1_score: float, report: dict):
    """Save evaluation results to CSV file."""
    csv_path = Path('out/evaluation_results.csv')
    csv_path.parent.mkdir(exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Checkpoint', 'Model', 'Test Scenario', 'F1 Score', 'Precision', 'Recall', 'Support'])
        
        # Extract metrics from report
        precision = report.get('macro avg', {}).get('precision', 0)
        recall = report.get('macro avg', {}).get('recall', 0)
        support = report.get('macro avg', {}).get('support', 0)
        
        # Write data
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            str(checkpoint_path),
            model_name,
            test_scenario,
            f"{f1_score:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            int(support)
        ])
    
    print(f"\nResults saved to: {csv_path}")


def predict_image(image_path: str, checkpoint_path: Path):
    """Predict the class of a single image using a saved checkpoint."""
    cfg = Config()
    
    # Extract model name from checkpoint path
    model_name = checkpoint_path.parent.name
    
    # Build model
    model = build_model(model_name, cfg.finetune_mode, cfg.drop_rate)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device())
    model.eval()  # Set model to evaluation mode
    
    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device())
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    
    print(f"Predicted class for image '{image_path}': {predicted_class}")
    return predicted_class


def run_batch_evaluation():
    """Run evaluation on multiple models and scenarios."""
    cfg = Config()
    out_dir = Path('out/head')
    
    print(f"Starting batch evaluation...")
    print(f"Models: {MODELS}")
    print(f"Test Scenarios: {TEST_SCENARIOS}")
    print(f"Seeds: {SEEDS}")
    print(f"Training Domain: {TRAIN_DOMAIN}")
    print("-" * 50)
    
    total_evaluations = 0
    successful_evaluations = 0
    
    for model_name in MODELS:
        model_dir = out_dir / model_name
        if not model_dir.exists():
            print(f"Warning: Model directory not found: {model_dir}")
            continue
            
        for seed in SEEDS:
            checkpoint_path = model_dir / f"train-mixed_{model_name}_seed{seed}_best.pt"
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found: {checkpoint_path}")
                continue
                
            for test_scenario in TEST_SCENARIOS:
                total_evaluations += 1
                try:
                    print(f"Evaluating: {model_name} | seed={seed} | scenario={test_scenario}")
                    evaluate_checkpoint(checkpoint_path, test_scenario)
                    successful_evaluations += 1
                except Exception as e:
                    print(f"Error evaluating {checkpoint_path} on {test_scenario}: {e}")
    
    print("-" * 50)
    print(f"Batch evaluation completed!")
    print(f"Successful: {successful_evaluations}/{total_evaluations}")
    print(f"Results saved to: out/evaluation_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate checkpoints or predict an image.")
    parser.add_argument('--predict', type=str, help="Path to the image for prediction.")
    parser.add_argument('--checkpoint', type=str, help="Path to the model checkpoint.")
    args = parser.parse_args()
    
    if args.predict and args.checkpoint:
        predict_image(args.predict, Path(args.checkpoint))
    else:
        run_batch_evaluation()

# Rice Leaf Disease Classification

This project implements deep learning models for rice leaf disease classification using computer vision techniques.



## Prerequisites

Make sure you have the `Dhan-Shomadhan` folder in the project root directory. This folder should contain the original rice leaf disease dataset with the following structure:

```
Dhan-Shomadhan/
├── Field Background/
│   ├── Brown Spot/
│   ├── Leaf Scald/
│   ├── Rice Blast/
│   ├── Rice Tungro/
│   └── Sheath Blight/
└── White Background/
    ├── Brown Spot/
    ├── Leaf Scald/
    ├── Rice Blast/
    ├── Rice Tungro/
    └── Sheath Blight/
```

## Installation

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Preview Mode (Test with 5 images)

Use this to quickly test CLAHE, mask, crop operations before processing the entire dataset:

```bash
python preprocess_offline.py --in_dir "./Dhan-Shomadhan" --out_dir "./preprocessed_mixed_preview" --mask --mask_mode robust --clahe --resize 224 --bg white --preview 5
```

### 2. Full Dataset Processing

After testing, remove the `--preview` flag to process all images:

```bash
python preprocess_offline.py --in_dir "./Dhan-Shomadhan" --out_dir "./preprocessed" --mask --mask_mode robust --clahe --resize 224 --bg white
```

### 3. Training

After preprocessing is complete, start training by running:

```bash
python run_experiments.py
```

## Configuration

You can modify which models are being trained by editing the `MODELS` variable in `compare.py`:

```python
MODELS = ['efficientnetb0','mobilenetv2']  # Add or remove models as needed
```

## Project Structure

- `Dhan-Shomadhan/`: Original dataset with rice leaf disease images
- `preprocessed/`: Processed dataset ready for training
- `out/`: Training outputs and model checkpoints
- `models/`: Model architecture definitions
- `utils/`: Utility functions for training and evaluation

## Features

- **Data Preprocessing**: CLAHE enhancement, green mask extraction, background removal
- **Multiple Models**: Support for ResNet50, EfficientNet, MobileNet architectures
- **Cross-domain Evaluation**: Training and testing on different background scenarios (white, field, mixed)
- **Data Augmentation**: Various augmentation techniques for improved generalization

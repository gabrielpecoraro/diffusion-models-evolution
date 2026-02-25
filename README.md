# PCB Defect Detection with Diffusion-Powered Data Augmentation

**Improving manufacturing quality control by generating synthetic defective PCB images with Stable Diffusion 1.5 + ControlNet, then training YOLOv8 to detect real defects more accurately.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

PCB (Printed Circuit Board) manufacturing defect detection is critical for quality control, but training robust detectors is difficult because **defects are rare**. Real-world datasets are small and class-imbalanced, leading to models that miss subtle defects like pinholes and mousebites.

## The Solution

This project uses **Stable Diffusion 1.5 + ControlNet (Canny edge conditioning)** to generate realistic synthetic defective PCB images that augment the training set. The key insight: ControlNet preserves the spatial layout of PCB traces while the diffusion model introduces realistic defect appearances via text prompts.

**Result**: Training YOLOv8 on real + synthetic data improves mAP by **+2-6%** over real-data-only baselines (consistent with [2024 published results](https://doi.org/10.3390/s24010268)).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    4-Stage Pipeline                              │
├─────────────┬───────────────┬───────────────┬───────────────────┤
│  1. DATA    │  2. GENERATE  │  3. DETECT    │  4. DEMO          │
│             │               │               │                   │
│ DeepPCB     │ PCB Template  │ Train A:      │ Gradio App        │
│ Dataset     │     │         │  Real only    │  - Upload PCB     │
│     │       │  Canny Edges  │     │         │  - Detect defects │
│ YOLO Format │     │         │ Train B:      │  - View results   │
│     │       │ SD1.5+ControlNet  Real+Synth │                   │
│ 80/10/10    │     │         │     │         │                   │
│  Split      │ Synthetic     │ Compare mAP   │                   │
│             │ Defect Images │               │                   │
└─────────────┴───────────────┴───────────────┴───────────────────┘
```

### Stage 1: Data — DeepPCB Dataset
- **1,500 image pairs** (template + defective), 640x640 resolution
- **6 defect classes**: open, short, mousebite, spur, copper, pinhole
- Automatic download, format conversion (DeepPCB → YOLO), and 80/10/10 split

### Stage 2: Generate — Synthetic Defect Images
- Extract **Canny edges** from PCB templates (captures trace layout)
- Feed edges to **ControlNet** + defect-specific text prompts → synthetic defective images
- **Annotation transfer**: bounding boxes from originals apply to synthetics (ControlNet preserves spatial structure)
- SSIM-based quality filtering removes failed generations
- Target: ~1,000 synthetic images

### Stage 3: Detect — YOLOv8 A/B Experiment
- **Experiment A** (baseline): Train YOLOv8s on real data only
- **Experiment B** (augmented): Train YOLOv8s on real + synthetic data
- Compare mAP50, mAP50-95, per-class precision/recall, confusion matrices

### Stage 4: Demo — Interactive Gradio App
- **Tab 1**: Upload a PCB image → detection overlay with bounding boxes
- **Tab 2**: Visualize the generation pipeline (template → edges → synthetic)
- **Tab 3**: View A/B comparison results

---

## Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM (Apple Silicon MPS or NVIDIA CUDA)
- [Hugging Face account](https://huggingface.co) for model downloads

### Setup

```bash
# Clone the repository
git clone https://github.com/gabrielpecoraro/diffusion-models-evolution.git
cd diffusion-models-evolution

# Create environment
conda create -n pcb-defect python=3.10 -y
conda activate pcb-defect

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for SD 1.5 + ControlNet)
huggingface-cli login
```

### Run the Full Pipeline

```bash
# Option 1: Run everything
make all

# Option 2: Run step by step
make download        # Download DeepPCB + convert to YOLO format
make generate        # Generate synthetic images with ControlNet
make train           # Train both YOLOv8 models (A/B experiment)
make evaluate        # Compare models and generate reports
make demo            # Launch Gradio web app
```

### Run Individual Scripts

```bash
# Download and prepare data
python scripts/download_data.py

# Generate synthetic images (adjust count as needed)
python scripts/generate_synthetic.py --num-images 1000

# Train detector (real-only, augmented, or both)
python scripts/train_detector.py --mode both --epochs 100

# Evaluate and compare
python scripts/evaluate.py

# Launch demo
python scripts/launch_demo.py --port 7860
```

---

## Project Structure

```
pcb-defect-diffusion/
├── configs/
│   ├── base.py                  # Project paths, seeds, device settings
│   ├── generation.py            # SD 1.5 + ControlNet parameters
│   └── detection.py             # YOLOv8 hyperparameters
│
├── src/
│   ├── data/
│   │   ├── download.py          # DeepPCB dataset downloader
│   │   ├── convert.py           # DeepPCB → YOLO format conversion
│   │   ├── split.py             # Train/val/test splits
│   │   └── inspect.py           # Dataset statistics and visualizations
│   │
│   ├── generation/
│   │   ├── pipeline.py          # SD 1.5 + ControlNet loader (MPS optimized)
│   │   ├── edge_extraction.py   # Canny edge maps for ControlNet
│   │   ├── prompt_bank.py       # Defect-specific text prompts (6 classes)
│   │   ├── generate.py          # Batch synthetic generation
│   │   └── postprocess.py       # SSIM-based quality filtering
│   │
│   ├── detection/
│   │   ├── trainer.py           # YOLOv8 training (real vs augmented)
│   │   ├── evaluate.py          # mAP, per-class metrics, confusion matrix
│   │   ├── predict.py           # Single-image inference
│   │   └── compare.py           # A/B comparison plots and reports
│   │
│   └── utils/
│       ├── memory.py            # MPS/CUDA memory management
│       ├── device.py            # Device and dtype detection
│       ├── logger.py            # Structured logging
│       └── visualization.py     # Shared plotting helpers
│
├── scripts/                     # CLI entry points
├── app/demo.py                  # Gradio web application
├── tests/                       # Unit tests (pytest)
├── Makefile                     # Pipeline automation
└── requirements.txt
```

---

## Technical Details

### Memory Budget (16GB Apple Silicon)

| Component | Memory (float16) | When Active |
|-----------|-----------------|-------------|
| SD 1.5 + ControlNet Canny | ~4-5 GB | Generation stage only |
| YOLOv8s training | ~2-3 GB | Detection stage only |
| YOLOv8s inference | ~0.5 GB | Demo stage only |

Models are never loaded simultaneously. Memory is cleared between stages.

### Key Design Decisions

1. **SD 1.5 over SDXL/FLUX**: 4GB vs 10-24GB. All 2024 PCB generation papers use SD 1.5. Mature ControlNet support.
2. **Canny edge conditioning**: Preserves PCB trace layout while allowing defect appearance variation.
3. **Annotation transfer** (not re-annotation): ControlNet preserves spatial structure → original bounding boxes apply to synthetics.
4. **YOLOv8s** (not nano): Better accuracy for small defects (pinholes, mousebites), still only 2-3GB.
5. **SSIM quality filtering**: Rejects synthetic images that are too similar (no augmentation value) or too different (failed generation).

### DeepPCB Defect Classes

| ID | Defect | Description |
|----|--------|-------------|
| 0 | Open | Broken/interrupted copper trace |
| 1 | Short | Unwanted copper bridge between traces |
| 2 | Mousebite | Irregular gap along trace edge |
| 3 | Spur | Small unwanted copper protrusion |
| 4 | Copper | Residual copper in etched area |
| 5 | Pinhole | Tiny void in copper fill |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_configs.py -v
pytest tests/test_data_convert.py -v
pytest tests/test_edge_extraction.py -v
```

---

## References

- Ding, R. et al. (2024). "Diffusion Model-Based Data Augmentation for PCB Defect Detection." *Sensors*, 24(1), 268. [DOI: 10.3390/s24010268](https://doi.org/10.3390/s24010268)
- [DeepPCB Dataset](https://github.com/tangsanli5201/DeepPCB) — Tang et al.
- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) — Rombach et al.
- [ControlNet](https://github.com/lllyasviel/ControlNet) — Zhang et al.
- [YOLOv8](https://github.com/ultralytics/ultralytics) — Ultralytics

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built by Gabriel Pecoraro*

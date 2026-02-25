# PCB Defect Detection — Diffusion-Powered Augmentation

## Project Overview
Real-world ML/CV application: generate synthetic PCB defect images with SD 1.5 + ControlNet Canny,
then train YOLOv8 to prove augmentation improves mAP. Targets 16GB Apple Silicon.

## Key Constraints
- 16GB Apple Silicon: always use CPU offloading, float16
- SD 1.5 + ControlNet Canny: ~4-5GB total (fits easily)
- YOLOv8s: ~2-3GB during training
- Never load generation and detection models simultaneously
- clear_memory() between pipeline stages

## 4-Stage Pipeline
1. DATA: Download DeepPCB → convert to YOLO format → train/val/test split
2. GENERATE: SD 1.5 + ControlNet Canny → synthetic defect images
3. DETECT: YOLOv8s train on real-only vs real+synthetic → compare mAP
4. DEMO: Gradio app for interactive defect detection

## Testing
- `pytest tests/ -v` (no GPU needed)
- `make download` → `make generate` → `make train` → `make demo`

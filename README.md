# Diffusion Models Evolution: From DDPM to FLUX.1

A comprehensive exploration of diffusion model architectures from 2022-2024, with working demos of **Stable Diffusion 3 Medium** and **FLUX.1-schnell** optimized for Apple Silicon.

<p align="center">
  <img src="assets/comparison_grid.png" width="800"/>
</p>
<p align="center"><em>SD3 Medium vs FLUX.1-schnell — same prompts, same seed, 512x512</em></p>

---

## 2024 AI Milestones Showcased

| Innovation | What Changed | Why It Matters |
|-----------|-------------|----------------|
| **MMDiT Architecture** | UNet replaced by Multimodal Diffusion Transformer | Two-way text-image attention, better text rendering, scalable to 12B+ params |
| **Flow Matching** | DDPM replaced by Rectified Flow | Straight sampling trajectories — 4-28 steps instead of 50-1000 |
| **Guidance Distillation** | CFG replaced by single-pass inference | 2x fewer forward passes per step |
| **GGUF Quantization** | 4-bit weight compression | Run 12B models on 16GB consumer hardware |

## Architecture Evolution

```
SD 1.5 (2022)          SD3 Medium (2024)         FLUX.1 (2024)
─────────────          ─────────────────         ──────────────
860M params            2B params                 12B params
UNet denoiser          MMDiT denoiser            MMDiT denoiser
CLIP-L encoder         CLIP-L + CLIP-G + T5      CLIP-L + T5
ε-prediction           Flow Matching             Flow Matching
50 steps               28 steps                  4 steps (schnell)
Cross-attention        Joint attention           Joint attention
(one-way)              (two-way)                 + guidance distillation
```

## Key Results

| Model | Steps | Avg Time (512x512) | Peak Memory | Quantization |
|-------|-------|-------------------|-------------|-------------|
| SD3 Medium | 28 | ~45s | ~4.3 GB | float16, no T5 |
| FLUX.1-schnell | 4 | ~30s | ~10 GB | GGUF Q4_K_S |

*Benchmarked on Apple Silicon M-series, 16GB unified memory*

## Project Structure

```
diffusion-models-evolution/
├── config/default.py                 # DiffusionConfig dataclass
├── models/
│   ├── pipeline_factory.py           # Unified SD3 + FLUX loader with GGUF
│   ├── memory_utils.py               # MPS memory management
│   └── prompt_bank.py                # Curated benchmark prompts
├── notebooks/
│   ├── 01_diffusion_fundamentals     # Theory: DDPM, noise schedules, sampling
│   ├── 02_architecture_evolution     # UNet → DiT → MMDiT → FLUX
│   ├── 03_flow_matching              # Rectified Flow + toy 2D implementation
│   ├── 04_sd3_medium_demo            # SD3 Medium inference on 16GB
│   ├── 05_flux_schnell_demo          # FLUX.1-schnell with GGUF quantization
│   └── 06_visual_comparison          # Head-to-head comparison grid
├── app/gradio_app.py                 # Interactive web demo
├── scripts/
│   ├── generate.py                   # CLI: generate single image
│   ├── compare.py                    # CLI: run comparison suite
│   └── benchmark.py                  # CLI: performance profiling
└── tests/                            # Unit tests (no GPU needed)
```

## Quick Start

### 1. Setup

```bash
# Clone the repo
git clone https://github.com/gabrielpecoraro/diffusion-models-evolution.git
cd diffusion-models-evolution

# Create conda environment
conda create -n diffusion python=3.11 -y
conda activate diffusion

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for SD3 Medium)
huggingface-cli login
```

### 2. Generate an Image

```bash
# FLUX.1-schnell (4 steps, GGUF quantized)
python scripts/generate.py --model flux-schnell --prompt "A cat holding a sign that says Hello"

# SD3 Medium (28 steps)
python scripts/generate.py --model sd3-medium --prompt "A sunset over mountains"
```

### 3. Run Comparison

```bash
python scripts/compare.py
# → Saves comparison grid to assets/comparison_grid.png
```

### 4. Launch Interactive Demo

```bash
python scripts/launch_app.py
# → Opens Gradio UI at http://localhost:7860
```

### 5. Explore the Notebooks

```bash
jupyter notebook notebooks/
```

Start with `01_diffusion_fundamentals.ipynb` for theory, or jump to `05_flux_schnell_demo.ipynb` for hands-on generation.

## Notebooks Overview

| # | Notebook | GPU? | Description |
|---|----------|------|-------------|
| 01 | Diffusion Fundamentals | No | DDPM theory, noise schedules, sampling algorithms |
| 02 | Architecture Evolution | No | SD1 → SD2 → SDXL → SD3 → FLUX architecture comparison |
| 03 | Flow Matching | No | Rectified Flow theory + trainable 2D toy implementation |
| 04 | SD3 Medium Demo | Yes | Full inference demo with performance profiling |
| 05 | FLUX.1-schnell Demo | Yes | GGUF-quantized 12B model inference in 4 steps |
| 06 | Visual Comparison | Yes | Head-to-head comparison grid (the LinkedIn showcase) |

## Apple Silicon Optimization

Running 12B-parameter models on 16GB required several tricks:

- **GGUF Quantization**: Q4_K_S compresses FLUX.1 from ~24GB to ~6.8GB
- **T5-XXL Removal**: Dropping SD3's largest text encoder saves ~14GB
- **CPU Offloading**: `enable_model_cpu_offload()` pages model layers between CPU and GPU
- **MPS Tuning**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` prevents memory allocation limits
- **Sequential Loading**: Never load both models simultaneously; `clear_memory()` between loads

## Tech Stack

- **PyTorch** + **MPS** (Apple Metal backend)
- **HuggingFace Diffusers** — unified pipeline API
- **GGUF** — 4-bit model quantization
- **Gradio** — interactive web demo
- **Matplotlib** — visualization and comparison grids

## References

- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) (SD3, 2024)
- Black Forest Labs, [FLUX.1](https://blackforestlabs.ai/) (2024)
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (DiT, 2023)
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (2022)

## License

MIT License — see [LICENSE](LICENSE) for details.

# Diffusion Models Evolution

## Project Overview
ML/CV portfolio project showcasing the evolution from DDPM to FLUX.1 (2024 milestone).
Optimized for 16GB Apple Silicon Mac using GGUF quantization and CPU offloading.

## Key Constraints
- 16GB unified memory: always use CPU offloading, max 512x512 resolution
- FLUX.1-schnell: load via GGUF Q4_K_S quantization (~6.8GB)
- SD3 Medium: drop T5-XXL encoder (`text_encoder_3=None, tokenizer_3=None`)
- Never load both models simultaneously — clear memory between loads

## Structure
- `config/default.py` — Central DiffusionConfig dataclass
- `models/` — Pipeline factory, memory utils, prompt bank
- `notebooks/` — 01-03 theory (no GPU), 04-06 demos (GPU)
- `app/` — Gradio interactive demo
- `scripts/` — CLI tools (generate, compare, benchmark)

## Testing
- `pytest tests/ -v` for unit tests (no GPU needed)
- `python scripts/generate.py --model flux-schnell --prompt "test" --steps 1 --height 256 --width 256` for smoke test

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Central configuration for all diffusion model operations.

    Defaults are tuned for 16GB Apple Silicon Mac.
    """

    # Model selection
    model_name: str = "flux-schnell"  # sd3-medium | flux-schnell
    quantization: str = "Q4_K_S"  # none | Q4_K_S | Q6_K | Q8_0

    # Generation
    prompt: str = "A photorealistic astronaut riding a horse on Mars"
    negative_prompt: str = ""
    num_inference_steps: int = 4  # FLUX schnell default; SD3 uses 28
    guidance_scale: float = 0.0  # FLUX schnell uses 0.0; SD3 uses 7.0
    height: int = 512
    width: int = 512
    seed: int = 42

    # Memory optimization
    dtype: str = "float16"
    enable_cpu_offload: bool = True
    drop_t5_encoder: bool = True  # SD3-specific: skip T5-XXL to save ~14GB
    mps_high_watermark_ratio: float = 0.0  # 0.0 = no MPS memory limit

    # Paths
    output_dir: str = "outputs"
    model_cache_dir: str = "model_cache"

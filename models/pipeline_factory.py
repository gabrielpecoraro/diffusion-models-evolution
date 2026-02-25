"""Unified pipeline factory for loading SD3 Medium and FLUX.1-schnell.

Handles model selection, GGUF quantization, and 16GB memory optimization.
"""

import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    StableDiffusion3Pipeline,
)

from config.default import DiffusionConfig
from models.memory_utils import (
    clear_memory,
    get_device,
    get_torch_dtype,
    log_memory_usage,
    setup_mps_environment,
)


GGUF_VARIANTS = {
    "Q4_K_S": "flux1-schnell-Q4_K_S.gguf",
    "Q4_0": "flux1-schnell-Q4_0.gguf",
    "Q6_K": "flux1-schnell-Q6_K.gguf",
    "Q8_0": "flux1-schnell-Q8_0.gguf",
}


def load_pipeline(config: DiffusionConfig):
    """Load a diffusion pipeline based on config.

    Returns a ready-to-use diffusers pipeline with memory optimizations applied.
    """
    setup_mps_environment(config.mps_high_watermark_ratio)
    dtype = get_torch_dtype(config.dtype)

    if config.model_name == "sd3-medium":
        pipe = _load_sd3(config, dtype)
    elif config.model_name == "flux-schnell":
        pipe = _load_flux_schnell(config, dtype)
    else:
        raise ValueError(
            f"Unknown model: {config.model_name}. "
            f"Supported: sd3-medium, flux-schnell"
        )

    log_memory_usage(f"after loading {config.model_name}")
    return pipe


def _load_sd3(config: DiffusionConfig, dtype: torch.dtype):
    """Load SD3 Medium with T5-XXL dropped to fit 16GB."""
    kwargs = {"torch_dtype": dtype}

    if config.drop_t5_encoder:
        # Dropping T5-XXL saves ~14GB; CLIP-L + CLIP-G still provide good text understanding
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        **kwargs,
    )

    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(get_device())

    return pipe


def _load_flux_schnell(config: DiffusionConfig, dtype: torch.dtype):
    """Load FLUX.1-schnell with optional GGUF quantization for 16GB."""
    if config.quantization != "none":
        gguf_file = GGUF_VARIANTS.get(config.quantization, GGUF_VARIANTS["Q4_K_S"])
        ckpt_url = (
            f"https://huggingface.co/city96/FLUX.1-schnell-gguf/"
            f"blob/main/{gguf_file}"
        )

        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            torch_dtype=dtype,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )

    if config.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(get_device())

    return pipe


def generate_image(pipe, config: DiffusionConfig):
    """Generate a single image using the loaded pipeline."""
    generator = torch.Generator().manual_seed(config.seed)

    image = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt if config.negative_prompt else None,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        generator=generator,
    ).images[0]

    clear_memory()
    return image

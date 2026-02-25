"""Load Stable Diffusion 1.5 + ControlNet Canny pipeline.

Optimised for 16 GB Apple Silicon: uses float16 weights (~4-5 GB VRAM),
the UniPC multi-step scheduler for fast inference, and optional model
CPU-offloading to keep peak memory well within budget.
"""

import logging

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from configs.base import ProjectConfig
from configs.generation import GenerationConfig
from src.utils.memory import get_device, get_torch_dtype, setup_mps_environment

logger = logging.getLogger(__name__)


def load_generation_pipeline(
    gen_config: GenerationConfig,
    project_config: ProjectConfig,
) -> StableDiffusionControlNetPipeline:
    """Load SD 1.5 + ControlNet Canny ready for inference.

    Steps
    -----
    1. Configure the MPS allocator (Apple Silicon high-watermark).
    2. Load the ControlNet Canny checkpoint in the project dtype.
    3. Load the Stable Diffusion 1.5 checkpoint with the ControlNet
       attached and the safety checker disabled (PCB images are
       industrial, never NSFW).
    4. Swap the scheduler to UniPC for higher quality at low step counts.
    5. Either enable sequential CPU-offloading (safest for 16 GB) or move
       the whole pipeline to the resolved device.

    Args:
        gen_config: Generation hyper-parameters (model IDs, etc.).
        project_config: Project-wide settings (device, dtype, offloading).

    Returns:
        A ready-to-call ``StableDiffusionControlNetPipeline``.
    """
    # --- 1. MPS environment ------------------------------------------------
    setup_mps_environment(project_config.mps_high_watermark_ratio)
    dtype: torch.dtype = get_torch_dtype(project_config.dtype)

    logger.info(
        "Loading ControlNet from '%s' (dtype=%s) ...",
        gen_config.controlnet_model,
        dtype,
    )

    # --- 2. ControlNet Canny -----------------------------------------------
    controlnet = ControlNetModel.from_pretrained(
        gen_config.controlnet_model,
        torch_dtype=dtype,
    )

    logger.info(
        "Loading Stable Diffusion from '%s' ...",
        gen_config.sd_model,
    )

    # --- 3. SD 1.5 pipeline ------------------------------------------------
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        gen_config.sd_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # --- 4. Fast scheduler --------------------------------------------------
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
    )

    # --- 5. Device placement ------------------------------------------------
    if project_config.enable_cpu_offload:
        logger.info("Enabling sequential CPU offload for memory safety.")
        pipe.enable_model_cpu_offload()
    else:
        device: torch.device = get_device()
        logger.info("Moving full pipeline to %s.", device)
        pipe = pipe.to(device)

    logger.info("Generation pipeline ready.")
    return pipe

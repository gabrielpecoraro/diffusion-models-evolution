"""Configuration for the synthetic data generation pipeline (SD 1.5 + ControlNet)."""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    # Model identifiers
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"

    # Canny edge detection
    canny_low: int = 100
    canny_high: int = 200

    # Generation parameters
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    height: int = 640
    width: int = 640

    # Batch generation
    num_synthetic_per_image: int = 3
    target_total_synthetic: int = 1000

    # Quality filtering
    min_ssim_to_original: float = 0.3
    max_ssim_to_original: float = 0.85

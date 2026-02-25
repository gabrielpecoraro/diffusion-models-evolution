"""Gradio web UI for comparing SD3 Medium and FLUX.1-schnell generations.

Launch with: python scripts/launch_app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch

from config.default import DiffusionConfig
from models.memory_utils import clear_memory, log_memory_usage
from models.pipeline_factory import generate_image, load_pipeline

# Global pipeline cache to avoid reloading on every generation
_current_pipeline = None
_current_model = None


def _get_pipeline(model_name):
    """Load pipeline, reusing cached one if same model."""
    global _current_pipeline, _current_model

    if _current_model == model_name and _current_pipeline is not None:
        return _current_pipeline

    # Unload previous model
    if _current_pipeline is not None:
        del _current_pipeline
        _current_pipeline = None
        _current_model = None
        clear_memory()

    config = DiffusionConfig(model_name=model_name)
    _current_pipeline = load_pipeline(config)
    _current_model = model_name
    return _current_pipeline


def generate_single(model_name, prompt, steps, guidance, seed, height, width):
    """Generate a single image from the selected model."""
    model_key = "flux-schnell" if "FLUX" in model_name else "sd3-medium"

    config = DiffusionConfig(
        model_name=model_key,
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        seed=int(seed),
        height=int(height),
        width=int(width),
    )

    pipe = _get_pipeline(model_key)

    start = time.perf_counter()
    image = generate_image(pipe, config)
    elapsed = time.perf_counter() - start

    mem = log_memory_usage("generation")
    info = (
        f"Model: {model_key} | Steps: {config.num_inference_steps} | "
        f"Time: {elapsed:.1f}s | Memory: {mem['rss_gb']:.1f} GB"
    )

    # Save
    out_dir = os.path.join("outputs", model_key.replace("-", "_"))
    os.makedirs(out_dir, exist_ok=True)
    image.save(os.path.join(out_dir, f"{config.seed}_{config.num_inference_steps}steps.png"))

    return image, info


def compare_models(prompt, seed):
    """Generate the same prompt with both models for comparison."""
    global _current_pipeline, _current_model

    # Unload any cached pipeline
    if _current_pipeline is not None:
        del _current_pipeline
        _current_pipeline = None
        _current_model = None
        clear_memory()

    # SD3 Medium
    sd3_config = DiffusionConfig(
        model_name="sd3-medium",
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        seed=int(seed),
        height=512, width=512,
    )
    pipe_sd3 = load_pipeline(sd3_config)
    start = time.perf_counter()
    sd3_image = generate_image(pipe_sd3, sd3_config)
    sd3_time = time.perf_counter() - start

    del pipe_sd3
    clear_memory()

    # FLUX.1-schnell
    flux_config = DiffusionConfig(
        model_name="flux-schnell",
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=int(seed),
        height=512, width=512,
    )
    pipe_flux = load_pipeline(flux_config)
    start = time.perf_counter()
    flux_image = generate_image(pipe_flux, flux_config)
    flux_time = time.perf_counter() - start

    del pipe_flux
    _current_pipeline = None
    _current_model = None
    clear_memory()

    return sd3_image, flux_image, f"SD3: {sd3_time:.1f}s | FLUX: {flux_time:.1f}s"


def load_gallery():
    """Load all previously generated images."""
    images = []
    for root, _, files in os.walk("outputs"):
        for f in sorted(files):
            if f.endswith(".png"):
                images.append(os.path.join(root, f))
    return images


def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(
        title="Diffusion Models Evolution",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Diffusion Models Evolution: SD3 Medium vs FLUX.1-schnell")
        gr.Markdown("**Device:** Apple Silicon MPS | **Memory:** 16 GB Unified | **Quantization:** GGUF Q4_K_S")

        with gr.Tabs():
            # Tab 1: Generate
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_select = gr.Radio(
                            ["SD3 Medium", "FLUX.1-schnell"],
                            value="FLUX.1-schnell",
                            label="Model",
                        )
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            value="A photorealistic astronaut riding a horse on Mars",
                            lines=3,
                        )
                        with gr.Row():
                            steps_slider = gr.Slider(1, 50, value=4, step=1, label="Steps")
                            guidance_slider = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            seed_input = gr.Number(value=42, label="Seed", precision=0)
                            height_input = gr.Number(value=512, label="Height", precision=0)
                            width_input = gr.Number(value=512, label="Width", precision=0)
                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image", type="pil")
                        perf_info = gr.Textbox(label="Performance", interactive=False)

                # Update defaults when model changes
                def update_defaults(model):
                    if "FLUX" in model:
                        return 4, 0.0
                    return 28, 7.0

                model_select.change(
                    update_defaults, inputs=[model_select],
                    outputs=[steps_slider, guidance_slider],
                )

                generate_btn.click(
                    generate_single,
                    inputs=[model_select, prompt_input, steps_slider,
                            guidance_slider, seed_input, height_input, width_input],
                    outputs=[output_image, perf_info],
                )

            # Tab 2: Compare
            with gr.Tab("Compare"):
                compare_prompt = gr.Textbox(
                    label="Prompt (same for both models)",
                    value="A corgi holding a wooden sign that reads 'Hello World'",
                    lines=2,
                )
                compare_seed = gr.Number(value=42, label="Seed", precision=0)
                compare_btn = gr.Button("Compare Side-by-Side", variant="primary")
                compare_info = gr.Textbox(label="Timing", interactive=False)

                with gr.Row():
                    sd3_output = gr.Image(label="SD3 Medium (28 steps)", type="pil")
                    flux_output = gr.Image(label="FLUX.1-schnell (4 steps)", type="pil")

                compare_btn.click(
                    compare_models,
                    inputs=[compare_prompt, compare_seed],
                    outputs=[sd3_output, flux_output, compare_info],
                )

            # Tab 3: Gallery
            with gr.Tab("Gallery"):
                refresh_btn = gr.Button("Refresh Gallery")
                gallery = gr.Gallery(label="Generated Images", columns=4)
                refresh_btn.click(load_gallery, outputs=[gallery])

    return app

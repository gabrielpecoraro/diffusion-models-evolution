"""Launch the Gradio PCB defect detection demo.

Usage:
    python scripts/launch_demo.py
    python scripts/launch_demo.py --port 7861 --share
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from app.demo import create_app
from configs.base import ProjectConfig


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio PCB defect detection demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLOv8 model weights")
    args = parser.parse_args()

    config = ProjectConfig()
    output_dir = config.resolve_path(config.output_dir)

    # Auto-detect best model
    model_path = None
    if args.model:
        model_path = Path(args.model)
    else:
        # Prefer augmented model, fall back to real-only
        for experiment in ["real_synthetic", "real_only"]:
            for best in (output_dir / experiment).rglob("best.pt"):
                model_path = best
                break
            if model_path:
                break

    if not model_path or not model_path.exists():
        print("WARNING: No trained model found. Run 'python scripts/train_detector.py' first.")
        print("The demo will start but detection won't work until a model is available.")

    app = create_app(model_path)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

"""Launch the Gradio web demo.

Usage:
    python scripts/launch_app.py
    python scripts/launch_app.py --port 7861 --share
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gradio_app import create_app


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

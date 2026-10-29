"""
Launch the Interactive Gradio Web IDE for SocraticCoder-7B (Week 6 Deliverable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.demo.gradio_ide import launch_gradio_ide


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SocraticCoder-7B Interactive Gradio IDE.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional LoRA checkpoint path.")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve Gradio UI.")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link.")
    args = parser.parse_args()

    launch_gradio_ide(checkpoint_dir=args.checkpoint, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()


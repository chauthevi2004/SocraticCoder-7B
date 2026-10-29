"""
Stage 3A CLI: Supervised Fine-Tuning (SFT Warm-Start) on Rejection-Sampled Socratic Trajectories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.training.sft_trainer import SocraticSFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3A SFT Warm-Start.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast verification without GPU model weights.",
    )
    args = parser.parse_args()

    cfg = SocraticCoderConfig.from_yaml(args.config)
    trainer = SocraticSFTTrainer(cfg)
    summary = trainer.train(dry_run=args.dry_run)
    print(
        f"[Stage 3A SFT Complete] Status={summary['status']} | "
        f"Steps={summary['total_steps']} | Final Loss={summary['final_loss']} | "
        f"Checkpoint='{summary['checkpoint_dir']}'"
    )


if __name__ == "__main__":
    main()


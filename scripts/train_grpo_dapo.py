"""
Stage 3B CLI: Contextual State-Jump RL Training via GRPO + DAPO (with beta ablation support).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.training.grpo_dapo_trainer import StateJumpGRPODAPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3B GRPO/DAPO Verifiable RL.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override beta weight on Isomorphic Transfer Gain DeltaPass(c'_transfer, U(P')). Use 0.0 for ablation.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="grpo_dapo_full",
        help="Subdirectory tag inside output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast State-Jump RL rollouts using heuristic student simulator.",
    )
    args = parser.parse_args()

    cfg = SocraticCoderConfig.from_yaml(args.config)
    sft_ckpt = Path(cfg.output_dir) / "sft_checkpoint"
    trainer = StateJumpGRPODAPOTrainer(
        config=cfg,
        beta_override=args.beta,
        run_tag=args.run_tag,
    )
    summary = trainer.train(
        sft_checkpoint_dir=sft_ckpt if sft_ckpt.exists() else None,
        dry_run=args.dry_run,
    )
    final_m = summary.get("final_metrics", {})
    print(
        f"[Stage 3B GRPO/DAPO Complete] Run='{summary['run_tag']}' (beta={summary['beta']}) | "
        f"Steps={summary['total_steps']} | MeanReward={final_m.get('mean_reward', 0):.4f} | "
        f"TransferDelta={final_m.get('mean_delta_transfer', 0) * 100:.1f}% | "
        f"AdvLeakage={final_m.get('adversarial_leakage_rate', 0) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()


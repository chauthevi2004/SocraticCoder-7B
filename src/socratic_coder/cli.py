"""
Unified CLI entry point (`python3 -m socratic_coder.cli`) supporting stage-by-stage
or full end-to-end execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SocraticCoderConfig
from .data.dataset_builder import Stage1DatasetBuilder
from .demo.gradio_ide import launch_gradio_ide
from .evaluation.evaluator import SocraticBenchmarkEvaluator
from .evaluation.plotting import generate_publication_plots
from .training.grpo_dapo_trainer import StateJumpGRPODAPOTrainer
from .training.sft_trainer import SocraticSFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="SocraticCoder-7B Unified CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "sft", "rl", "ablation", "eval", "demo"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast verification without GPU model weights.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Build dataset using only verified local algorithmic bank.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Launch Gradio demo with a public share link.",
    )
    args = parser.parse_args()

    cfg = SocraticCoderConfig.from_yaml(args.config)

    if args.stage in ("all", "data"):
        builder = Stage1DatasetBuilder(
            seed=cfg.seed,
            max_dialogue_turns=cfg.dataset.max_dialogue_turns,
            adversarial_ratio=cfg.dataset.adversarial_ratio,
            timeout_sec=cfg.evaluation.test_timeout_sec,
        )
        train_tasks, test_tasks, sft_examples = builder.build_all(
            num_train=cfg.dataset.num_train_pairs,
            num_test=cfg.dataset.num_test_pairs,
            num_sft=cfg.dataset.num_sft_trajectories,
            output_dir=cfg.data_dir,
            try_load_hf=not (args.offline_only or args.dry_run),
        )
        print(
            f"[Stage 1 Data] Saved {len(train_tasks)} train pairs, "
            f"{len(test_tasks)} test pairs, {len(sft_examples)} SFT examples -> '{cfg.data_dir}'"
        )

    if args.stage in ("all", "sft"):
        sft_trainer = SocraticSFTTrainer(cfg)
        sft_summary = sft_trainer.train(dry_run=args.dry_run)
        print(
            f"[Stage 2 SFT] Status={sft_summary['status']} | "
            f"Steps={sft_summary['total_steps']} | Final Loss={sft_summary['final_loss']}"
        )

    sft_ckpt = Path(cfg.output_dir) / "sft_checkpoint"

    if args.stage in ("all", "rl"):
        rl_trainer = StateJumpGRPODAPOTrainer(
            config=cfg,
            beta_override=cfg.reward.beta,
            run_tag="grpo_dapo_full",
        )
        rl_summary = rl_trainer.train(
            sft_checkpoint_dir=sft_ckpt if sft_ckpt.exists() else None,
            dry_run=args.dry_run,
        )
        print(f"[Stage 3A RL (beta={cfg.reward.beta})] Complete -> '{rl_summary['checkpoint_dir']}'")

    if args.stage in ("all", "ablation"):
        ablation_steps = max(5, cfg.rl.num_steps // 2) if not args.dry_run else cfg.rl.num_steps
        rl_ablation = StateJumpGRPODAPOTrainer(
            config=cfg,
            beta_override=0.0,
            run_tag="grpo_dapo_beta0",
        )
        ab_summary = rl_ablation.train(
            sft_checkpoint_dir=sft_ckpt if sft_ckpt.exists() else None,
            dry_run=args.dry_run,
            num_steps_override=ablation_steps,
        )
        print(f"[Stage 3B Ablation (beta=0.0)] Complete -> '{ab_summary['checkpoint_dir']}'")

    if args.stage in ("all", "eval"):
        evaluator = SocraticBenchmarkEvaluator(cfg)
        report = evaluator.evaluate_all(dry_run=args.dry_run)
        print("\n" + report.markdown_table)

        rl_full_path = Path(cfg.output_dir) / "grpo_dapo_full" / "rl_training_log.json"
        rl_b0_path = Path(cfg.output_dir) / "grpo_dapo_beta0" / "rl_training_log.json"
        rl_full_log = (
            json.loads(rl_full_path.read_text(encoding="utf-8")) if rl_full_path.exists() else None
        )
        rl_b0_log = (
            json.loads(rl_b0_path.read_text(encoding="utf-8")) if rl_b0_path.exists() else None
        )
        figs = generate_publication_plots(
            report=report,
            rl_full_log=rl_full_log,
            rl_beta0_log=rl_b0_log,
            output_dir=Path(cfg.results_dir) / "figures",
        )
        print(f"[Stage 4 Eval] Saved reports to '{cfg.results_dir}' and figures: {figs}")

    if args.stage == "demo":
        ckpt_dir = Path(cfg.output_dir) / "grpo_dapo_full"
        launch_gradio_ide(
            checkpoint_dir=str(ckpt_dir) if ckpt_dir.exists() else None,
            share=args.share,
        )


if __name__ == "__main__":
    main()

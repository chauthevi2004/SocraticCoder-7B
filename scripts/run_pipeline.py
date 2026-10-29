"""
One-Command End-to-End Orchestrator for SocraticCoder-7B.

Executes all 4 stages sequentially and saves all checkpoints, evaluation tables,
and publication plots:
  1. Stage 1: Paired Isomorphic Problem (`P, P'`) & Misconception Environment Builder
  2. Stage 3A: Supervised Fine-Tuning (SFT Warm-Start)
  3. Stage 3B (Main): Contextual State-Jump RL via GRPO + DAPO (`beta = 0.60`)
  4. Stage 3B (Ablation): Practice-Only RL (`beta = 0.0`) to quantify transfer reward impact
  5. Stage 4: Full Evaluation across Student Models + Adversarial Leakage + Plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.data.dataset_builder import Stage1DatasetBuilder
from socratic_coder.evaluation.evaluator import SocraticBenchmarkEvaluator
from socratic_coder.evaluation.plotting import generate_publication_plots
from socratic_coder.training.grpo_dapo_trainer import StateJumpGRPODAPOTrainer
from socratic_coder.training.sft_trainer import SocraticSFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-End SocraticCoder-7B Pipeline (Data -> SFT -> GRPO/DAPO -> Ablation -> Eval)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML config (e.g. configs/kaggle_t4x2.yaml or configs/full_1200.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast end-to-end verification using real unit-test verifiers + heuristic student simulator.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Build dataset using verified local algorithmic bank without querying HuggingFace Hub.",
    )
    args = parser.parse_args()

    t0 = time.time()
    cfg = SocraticCoderConfig.from_yaml(args.config)
    print(f"======================================================================")
    print(f" Starting {cfg.project_name} End-to-End Pipeline")
    print(f" Config: {args.config} | Mode: {'DRY-RUN / FAST VERIFY' if args.dry_run else 'FULL GPU TRAINING'}")
    print(f"======================================================================")

    # 1. Stage 1: Build Dataset
    print("\n[1/5] Building Stage 1 Paired Isomorphic Coding Dataset (P, P') & SFT Trajectories...")
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
        f"  -> Built {len(train_tasks)} train pairs, {len(test_tasks)} test pairs, "
        f"{len(sft_examples)} gate-verified SFT trajectories."
    )

    # 2. Stage 3A: SFT Warm-Start
    print("\n[2/5] Running Stage 3A Supervised Fine-Tuning (SFT Warm-Start)...")
    sft_trainer = SocraticSFTTrainer(cfg)
    sft_summary = sft_trainer.train(dry_run=args.dry_run)
    print(
        f"  -> SFT Complete: steps={sft_summary['total_steps']}, "
        f"final_loss={sft_summary['final_loss']}"
    )

    # 3. Stage 3B: Main GRPO/DAPO Training (beta = 0.60)
    print(f"\n[3/5] Running Stage 3B Contextual State-Jump RL (GRPO + DAPO, beta={cfg.reward.beta})...")
    rl_trainer_full = StateJumpGRPODAPOTrainer(
        config=cfg,
        beta_override=cfg.reward.beta,
        run_tag="grpo_dapo_full",
    )
    rl_full_summary = rl_trainer_full.train(
        sft_checkpoint_dir=sft_summary["checkpoint_dir"],
        dry_run=args.dry_run,
    )
    fm_full = rl_full_summary["final_metrics"]
    print(
        f"  -> Full GRPO/DAPO Complete: mean_reward={fm_full.get('mean_reward', 0):.4f}, "
        f"transfer_delta=+{fm_full.get('mean_delta_transfer', 0) * 100:.1f}%, "
        f"adv_leakage={fm_full.get('adversarial_leakage_rate', 0) * 100:.1f}%"
    )

    # 4. Stage 3B Ablation: Practice-Only RL (beta = 0.0)
    print("\n[4/5] Running Stage 3B Ablation Study (Practice-Only RL: beta=0.0)...")
    ablation_steps = max(5, cfg.rl.num_steps // 2) if not args.dry_run else cfg.rl.num_steps
    rl_trainer_b0 = StateJumpGRPODAPOTrainer(
        config=cfg,
        beta_override=0.0,
        run_tag="grpo_dapo_beta0",
    )
    rl_b0_summary = rl_trainer_b0.train(
        sft_checkpoint_dir=sft_summary["checkpoint_dir"],
        dry_run=args.dry_run,
        num_steps_override=ablation_steps,
    )
    fm_b0 = rl_b0_summary["final_metrics"]
    print(
        f"  -> Ablation (beta=0.0) Complete: transfer_delta=+{fm_b0.get('mean_delta_transfer', 0) * 100:.1f}%, "
        f"adv_leakage={fm_b0.get('adversarial_leakage_rate', 0) * 100:.1f}%"
    )

    # 5. Stage 4: Evaluation & Publication Plots
    print("\n[5/5] Running Stage 4 Multi-Student Evaluation & Generating Figures...")
    evaluator = SocraticBenchmarkEvaluator(cfg)
    report = evaluator.evaluate_all(dry_run=args.dry_run)
    figs = generate_publication_plots(
        report=report,
        rl_full_log=rl_full_summary,
        rl_beta0_log=rl_b0_summary,
        output_dir=Path(cfg.results_dir) / "figures",
    )

    print("\n" + report.markdown_table)
    print(f"Generated figures: {figs}")
    print(f"Total pipeline elapsed time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()


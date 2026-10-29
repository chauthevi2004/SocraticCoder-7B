"""
Stage 4 CLI: Run Full Benchmark Evaluation, Ablation Comparison (beta=0 vs beta=0.60),
and Generate High-Resolution Publication Figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.evaluation.evaluator import SocraticBenchmarkEvaluator
from socratic_coder.evaluation.plotting import generate_publication_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 Evaluation & Ablation Suite.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast deterministic evaluation.",
    )
    args = parser.parse_args()

    cfg = SocraticCoderConfig.from_yaml(args.config)
    evaluator = SocraticBenchmarkEvaluator(cfg)
    report = evaluator.evaluate_all(dry_run=args.dry_run)

    print("\n" + report.markdown_table)

    rl_full_path = Path(cfg.output_dir) / "grpo_dapo_full" / "rl_training_log.json"
    rl_b0_path = Path(cfg.output_dir) / "grpo_dapo_beta0" / "rl_training_log.json"
    rl_full_log = json.loads(rl_full_path.read_text(encoding="utf-8")) if rl_full_path.exists() else None
    rl_b0_log = json.loads(rl_b0_path.read_text(encoding="utf-8")) if rl_b0_path.exists() else None

    figs = generate_publication_plots(
        report=report,
        rl_full_log=rl_full_log,
        rl_beta0_log=rl_b0_log,
        output_dir=Path(cfg.results_dir) / "figures",
    )
    print(f"[Stage 4 Complete] Saved JSON/Markdown reports to '{cfg.results_dir}' and {len(figs)} figures: {figs}")


if __name__ == "__main__":
    main()


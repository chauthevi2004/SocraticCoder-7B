"""
Stage 1 CLI: Build and verify the Paired Isomorphic Coding Dataset (P, P')
and rejection-sampled Socratic SFT trajectories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.data.dataset_builder import Stage1DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 1 Paired Isomorphic Dataset (P, P').")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_t4x2.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use only the verified local algorithmic seed bank without querying HuggingFace Hub.",
    )
    args = parser.parse_args()

    cfg = SocraticCoderConfig.from_yaml(args.config)
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
        try_load_hf=not args.offline_only,
    )
    print(
        f"[Stage 1 Complete] Saved {len(train_tasks)} train pairs, "
        f"{len(test_tasks)} held-out test pairs, and {len(sft_examples)} SFT trajectories "
        f"to '{cfg.data_dir}'."
    )


if __name__ == "__main__":
    main()


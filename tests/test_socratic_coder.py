"""
Comprehensive Unit & Integration Tests for SocraticCoder-7B.

Run with either:
  PYTHONPATH=src python3 -m unittest discover -s tests -v
or:
  pytest -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.data.algorithmic_bank import get_seed_paired_problems
from socratic_coder.data.dataset_builder import Stage1DatasetBuilder
from socratic_coder.env.student_simulator import (
    StateJumpTutorEnvironment,
    StudentSimulator,
)
from socratic_coder.evaluation.evaluate import EvaluationSuite
from socratic_coder.training.grpo_dapo_trainer import (
    StateJumpGRPODAPOTrainer,
    compute_group_advantages,
)
from socratic_coder.training.sft_trainer import SocraticSFTTrainer
from socratic_coder.verifiers.code_executor import SandboxedCodeVerifier
from socratic_coder.verifiers.leakage_gate import ExecutableNonLeakageGate
from socratic_coder.verifiers.reward import CompositePedagogicalReward


class TestExecutableNonLeakageGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ExecutableNonLeakageGate(
            max_code_lines=2,
            require_diagnostic_question=True,
            max_ngram_overlap=0.35,
            ngram_size=4,
        )
        self.buggy = (
            "def min_coins(coins, amount):\n"
            "    dp = [float('inf')] * amount\n"
            "    return dp[amount - 1]\n"
        )
        self.canonical = (
            "def min_coins(coins, amount):\n"
            "    dp = [float('inf')] * (amount + 1)\n"
            "    return dp[amount] if dp[amount] != float('inf') else -1\n"
        )

    def test_valid_socratic_hint_passes(self) -> None:
        hint = (
            "Think about what index in your DP table represents the target value N when "
            "index 0 represents zero. How many slots do you need to index up to N inclusive?"
        )
        verdict = self.gate.verify(hint, self.buggy, self.canonical)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.gate_value, 1)
        self.assertEqual(verdict.reasons, [])

    def test_fenced_code_dump_blocked(self) -> None:
        leaky = (
            "Here is how you should write it:\n"
            "```python\n"
            "dp = [float('inf')] * (amount + 1)\n"
            "dp[0] = 0\n"
            "return dp[amount]\n"
            "```\n"
            "Does that make sense?"
        )
        verdict = self.gate.verify(leaky, self.buggy, self.canonical)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate_value, 0)
        self.assertTrue(any("fenced_code_exceeds_limit" in r for r in verdict.reasons))

    def test_missing_diagnostic_question_blocked(self) -> None:
        declarative = (
            "Your DP table allocation is off by one element because you need to include zero."
        )
        verdict = self.gate.verify(declarative, self.buggy, self.canonical)
        self.assertFalse(verdict.passed)
        self.assertIn("lacks_diagnostic_question", verdict.reasons)

    def test_high_ngram_patch_overlap_blocked(self) -> None:
        patch_leak = (
            "Why not just write `return dp[amount] if dp[amount] != float('inf') else -1`?"
        )
        verdict = self.gate.verify(patch_leak, self.buggy, self.canonical)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("patch_ngram_overlap_exceeded" in r for r in verdict.reasons))


class TestSeedBankAndSandboxedVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = SandboxedCodeVerifier(timeout_sec=2.0, max_workers=2)
        self.gate = ExecutableNonLeakageGate()

    def test_all_seed_problems_verified(self) -> None:
        seeds = get_seed_paired_problems()
        self.assertGreaterEqual(len(seeds), 10)

        for item in seeds:
            fid = item["family_id"]
            p_canon = self.verifier.evaluate(
                item["p_canonical_solution"], item["p_test_code"], item["p_entry_point"]
            )
            self.assertEqual(
                p_canon.pass_rate,
                1.0,
                f"Practice canonical solution failed for {fid}: {p_canon.error_message}",
            )

            p_buggy = self.verifier.evaluate(
                item["p_buggy_code"], item["p_test_code"], item["p_entry_point"]
            )
            self.assertLess(
                p_buggy.pass_rate,
                1.0,
                f"Practice buggy code unexpectedly passed 100% for {fid}",
            )

            pp_canon = self.verifier.evaluate(
                item["p_prime_canonical_solution"],
                item["p_prime_test_code"],
                item["p_prime_entry_point"],
            )
            self.assertEqual(
                pp_canon.pass_rate,
                1.0,
                f"Transfer P' canonical solution failed for {fid}: {pp_canon.error_message}",
            )

            gate_verdict = self.gate.verify(
                item["gold_socratic_hint"],
                item["p_buggy_code"],
                item["p_canonical_solution"],
            )
            self.assertTrue(
                gate_verdict.passed,
                f"Gold Socratic hint failed non-leakage gate for {fid}: {gate_verdict.reasons}",
            )

    def test_timeout_and_syntax_error_handling(self) -> None:
        bad_syntax = "def foo(x)\n    return x"
        res_syn = self.verifier.evaluate(bad_syntax, "assert foo(1) == 1", "foo")
        self.assertFalse(res_syn.syntax_valid)
        self.assertEqual(res_syn.pass_rate, 0.0)

        fast_verifier = SandboxedCodeVerifier(timeout_sec=0.4, max_workers=1)
        inf_loop = "def foo(x):\n    while True:\n        pass\n"
        res_timeout = fast_verifier.evaluate(inf_loop, "assert foo(1) == 1", "foo")
        self.assertTrue(res_timeout.timed_out)
        self.assertEqual(res_timeout.pass_rate, 0.0)


class TestStateJumpRLAndEndToEndPipeline(unittest.TestCase):
    def test_dapo_group_advantages(self) -> None:
        adv_zero, _, _, is_zero = compute_group_advantages([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(is_zero)
        self.assertEqual(adv_zero, [0.0, 0.0, 0.0, 0.0])

        adv_valid, _, _, is_zero_2 = compute_group_advantages([0.0, 0.5, 0.8, 0.2])
        self.assertFalse(is_zero_2)
        self.assertAlmostEqual(sum(adv_valid), 0.0, places=5)

    def test_full_pipeline_smoke_run(self) -> None:
        cfg = SocraticCoderConfig.from_yaml("configs/fast_smoke_test.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg.data_dir = str(Path(tmp) / "data")
            cfg.output_dir = str(Path(tmp) / "outputs")
            cfg.results_dir = str(Path(tmp) / "results")

            builder = Stage1DatasetBuilder(seed=42)
            train_tasks, test_tasks, sft_trajs = builder.build_all(
                num_train=12,
                num_test=8,
                num_sft=12,
                output_dir=cfg.data_dir,
                try_load_hf=False,
            )
            self.assertEqual(len(train_tasks), 12)
            self.assertEqual(len(test_tasks), 8)
            self.assertEqual(len(sft_trajs), 12)

            sft_res = SocraticSFTTrainer(cfg).train(dry_run=True)
            self.assertEqual(sft_res["status"], "completed_dry_run")

            rl_res = StateJumpGRPODAPOTrainer(cfg, ablation_no_transfer=False).train(
                dry_run=True
            )
            self.assertIn("completed", rl_res["status"])
            self.assertGreater(rl_res["final_metrics"]["mean_reward"], 0.0)

            abl_res = StateJumpGRPODAPOTrainer(cfg, ablation_no_transfer=True).train(
                dry_run=True
            )
            self.assertIn("completed", abl_res["status"])

            eval_res = EvaluationSuite(cfg).run_all(dry_run=True)
            self.assertIn("markdown_table", eval_res)
            self.assertTrue((Path(cfg.results_dir) / "plots" / "ablation_comparison.svg").exists())


if __name__ == "__main__":
    unittest.main()

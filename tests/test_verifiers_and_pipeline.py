"""
Unit Tests for SocraticCoder-7B:
1. Executable Non-Leakage Gate I_no-leak(a_t) in {0, 1}
2. Sandboxed Deterministic Unit-Test Executor Pass(c, U(P)) in [0, 1]
3. Composite Pedagogical Reward R(a_t) & Gate-First Short-Circuiting
4. Stage 1 Paired Isomorphic Dataset Builder & Seed Bank Verification
5. Stage 3B GRPO/DAPO Group Advantage Normalization & Dynamic Sampling Detection
"""

from __future__ import annotations

from pathlib import Path

from socratic_coder.config import SocraticCoderConfig
from socratic_coder.data.algorithmic_bank import get_seed_paired_problems
from socratic_coder.data.dataset_builder import Stage1DatasetBuilder
from socratic_coder.env.student_simulator import StateJumpTutorEnvironment, StudentSimulator
from socratic_coder.training.grpo_dapo_trainer import compute_group_advantages
from socratic_coder.verifiers.code_executor import SandboxedCodeVerifier
from socratic_coder.verifiers.leakage_gate import ExecutableNonLeakageGate
from socratic_coder.verifiers.reward import CompositePedagogicalReward


def test_executable_non_leakage_gate() -> None:
    gate = ExecutableNonLeakageGate(
        max_code_lines=2,
        require_diagnostic_question=True,
        max_ngram_overlap=0.35,
    )
    seed = get_seed_paired_problems()[0]
    buggy = seed["p_buggy_code"]
    canon = seed["p_canonical_solution"]

    # 1. Gold Socratic hint should pass I_no-leak == 1
    verdict_ok = gate.verify(seed["gold_socratic_hint"], buggy, canon)
    assert verdict_ok.passed is True
    assert verdict_ok.gate_value == 1

    # 2. Full code dump in markdown fence should be blocked (I_no-leak == 0)
    leaky_code = f"Here is the fixed code? \n```python\n{canon}\n```"
    verdict_dump = gate.verify(leaky_code, buggy, canon)
    assert verdict_dump.passed is False
    assert verdict_dump.gate_value == 0

    # 3. Declarative statement without diagnostic Socratic question should be blocked
    no_question = "You need to allocate one extra element in the DP list and loop up to amount inclusive."
    verdict_no_q = gate.verify(no_question, buggy, canon)
    assert verdict_no_q.passed is False
    assert "lacks_diagnostic_question" in verdict_no_q.reasons

    # 4. High patch n-gram overlap (>35%) disguised in inline backticks should be blocked
    high_overlap = (
        "Why don't you just write `dp = [float('inf')] * (amount + 1)` and "
        "`for x in range(1, amount + 1):` and `return dp[amount] if dp[amount] != float('inf') else -1`?"
    )
    verdict_overlap = gate.verify(high_overlap, buggy, canon)
    assert verdict_overlap.passed is False
    assert verdict_overlap.gate_value == 0


def test_sandboxed_code_verifier_and_seed_bank() -> None:
    verifier = SandboxedCodeVerifier(timeout_sec=2.0, max_workers=2)
    seeds = get_seed_paired_problems()
    assert len(seeds) >= 10

    for item in seeds:
        res_canon_p = verifier.evaluate(
            item["p_canonical_solution"], item["p_test_code"], item["p_entry_point"]
        )
        res_buggy_p = verifier.evaluate(
            item["p_buggy_code"], item["p_test_code"], item["p_entry_point"]
        )
        res_canon_pp = verifier.evaluate(
            item["p_prime_canonical_solution"],
            item["p_prime_test_code"],
            item["p_prime_entry_point"],
        )
        assert res_canon_p.pass_rate == 1.0, f"Failed canonical P on {item['family_id']}"
        assert res_buggy_p.pass_rate < 1.0, f"Buggy P unexpectedly passed all tests on {item['family_id']}"
        assert res_canon_pp.pass_rate == 1.0, f"Failed canonical P' on {item['family_id']}"


def test_composite_reward_and_state_jump_env(tmp_path: Path) -> None:
    builder = Stage1DatasetBuilder(seed=42)
    train_tasks, test_tasks, sft_examples = builder.build_all(
        num_train=6,
        num_test=4,
        num_sft=6,
        output_dir=tmp_path / "data",
        try_load_hf=False,
    )
    assert len(train_tasks) == 6
    assert len(test_tasks) == 4
    assert all(ex.gate_passed for ex in sft_examples)

    task = train_tasks[0]
    gate = ExecutableNonLeakageGate()
    verifier = SandboxedCodeVerifier(timeout_sec=2.0, max_workers=2)
    reward_fn = CompositePedagogicalReward(
        leakage_gate=gate,
        code_verifier=verifier,
        alpha=0.40,
        beta=0.60,
        gamma=0.0005,
        gate_first_short_circuit=True,
    )
    student = StudentSimulator(backend="heuristic")
    env = StateJumpTutorEnvironment(student=student, reward_fn=reward_fn)

    candidates = [
        task.gold_socratic_hint,
        f"```python\n{task.p_canonical_solution}\n```",
    ]
    rollouts = env.evaluate_group(task, candidates)
    assert len(rollouts) == 2

    # Candidate 0 (Socratic hint) passes gate and achieves positive composite reward
    assert rollouts[0].reward_breakdown.gate_value == 1
    assert rollouts[0].reward_breakdown.total_reward > 0.3

    # Candidate 1 (Solution dump) is short-circuited by I_no-leak == 0 and gets 0.0 reward
    assert rollouts[1].reward_breakdown.gate_value == 0
    assert rollouts[1].reward_breakdown.total_reward == 0.0
    assert rollouts[1].reward_breakdown.short_circuited is True


def test_grpo_dapo_advantage_normalization() -> None:
    # Non-zero variance group
    adv, mean_r, std_r, is_zero = compute_group_advantages([0.0, 0.4, 0.8, 0.8])
    assert is_zero is False
    assert abs(sum(adv)) < 1e-5
    assert adv[-1] > adv[0]

    # Zero-variance group triggers DAPO dynamic resampling
    adv_z, _, _, is_zero_z = compute_group_advantages([0.0, 0.0, 0.0, 0.0])
    assert is_zero_z is True
    assert all(a == 0.0 for a in adv_z)


def test_yaml_configs_load_cleanly() -> None:
    for cfg_name in (
        "fast_smoke_test.yaml",
        "kaggle_t4x2.yaml",
        "full_1200.yaml",
    ):
        cfg = SocraticCoderConfig.from_yaml(Path("configs") / cfg_name)
        assert cfg.reward.alpha == 0.40
        assert cfg.reward.beta == 0.60
        assert cfg.leakage_gate.max_code_lines == 2


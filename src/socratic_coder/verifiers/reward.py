"""
Composite Pedagogical Reward R(a_t) for SocraticCoder-7B (Section 2.1 of Proposal):

    R(a_t) = I_no-leak(a_t) * (
        alpha * Pass(c_final, U(P))
        + beta * DeltaPass(c'_transfer, U(P'))
        - gamma * Tok(a_t)
    )

where:
- I_no-leak(a_t) in {0, 1} is the Executable Non-Leakage Gate
- Pass(c_final, U(P)) in [0, 1] is the fraction of unit tests passed on practice problem P
- DeltaPass(c'_transfer, U(P')) = Pass(c'_transfer, U(P')) - Pass_base(P') is the
  student's zero-shot transfer learning gain on unseen isomorphic problem P'
- Tok(a_t) is the token length of tutor hint a_t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .code_executor import ExecutionResult, SandboxedCodeVerifier
from .leakage_gate import ExecutableNonLeakageGate, LeakageVerdict


@dataclass
class RewardBreakdown:
    """Full diagnostic breakdown of R(a_t) for logging, GRPO/DAPO, and UI visualization."""

    total_reward: float
    gate_value: int
    gate_verdict: LeakageVerdict
    pass_practice: float
    pass_transfer: float
    baseline_transfer: float
    delta_pass_transfer: float
    token_count: int
    length_penalty: float
    short_circuited: bool = False
    practice_exec: Optional[ExecutionResult] = None
    transfer_exec: Optional[ExecutionResult] = None

    def to_dict(self) -> dict:
        return {
            "total_reward": round(self.total_reward, 4),
            "gate_value": self.gate_value,
            "gate_verdict": self.gate_verdict.to_dict(),
            "pass_practice": round(self.pass_practice, 4),
            "pass_transfer": round(self.pass_transfer, 4),
            "baseline_transfer": round(self.baseline_transfer, 4),
            "delta_pass_transfer": round(self.delta_pass_transfer, 4),
            "token_count": self.token_count,
            "length_penalty": round(self.length_penalty, 4),
            "short_circuited": self.short_circuited,
        }


class CompositePedagogicalReward:
    """
    Computes the verifiable pedagogical reward R(a_t) with Gate-First Short-Circuiting.
    """

    def __init__(
        self,
        leakage_gate: ExecutableNonLeakageGate,
        code_verifier: SandboxedCodeVerifier,
        alpha: float = 0.40,
        beta: float = 0.60,
        gamma: float = 0.0005,
        gate_first_short_circuit: bool = True,
    ) -> None:
        self.leakage_gate = leakage_gate
        self.code_verifier = code_verifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gate_first_short_circuit = gate_first_short_circuit

    @staticmethod
    def count_tokens(text: str, tokenizer=None) -> int:
        if not text:
            return 0
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fast whitespace + punctuation approximation when tokenizer is not passed
        return max(1, int(len(text.split()) * 1.3))

    def check_gate(
        self,
        tutor_response: str,
        buggy_code: str,
        canonical_solution: str,
    ) -> LeakageVerdict:
        return self.leakage_gate.verify(
            tutor_response=tutor_response,
            buggy_code=buggy_code,
            canonical_solution=canonical_solution,
        )

    def compute_from_executions(
        self,
        tutor_response: str,
        gate_verdict: LeakageVerdict,
        practice_exec: Optional[ExecutionResult],
        transfer_exec: Optional[ExecutionResult],
        baseline_transfer_pass: float = 0.0,
        tokenizer=None,
    ) -> RewardBreakdown:
        tok_count = self.count_tokens(tutor_response, tokenizer=tokenizer)
        length_penalty = self.gamma * tok_count

        if not gate_verdict.passed or gate_verdict.gate_value == 0:
            return RewardBreakdown(
                total_reward=0.0,
                gate_value=0,
                gate_verdict=gate_verdict,
                pass_practice=practice_exec.pass_rate if practice_exec else 0.0,
                pass_transfer=transfer_exec.pass_rate if transfer_exec else 0.0,
                baseline_transfer=baseline_transfer_pass,
                delta_pass_transfer=(
                    (transfer_exec.pass_rate - baseline_transfer_pass)
                    if transfer_exec
                    else 0.0
                ),
                token_count=tok_count,
                length_penalty=length_penalty,
                short_circuited=(practice_exec is None and transfer_exec is None),
                practice_exec=practice_exec,
                transfer_exec=transfer_exec,
            )

        pass_p = practice_exec.pass_rate if practice_exec is not None else 0.0
        pass_p_prime = transfer_exec.pass_rate if transfer_exec is not None else 0.0
        delta_pass = pass_p_prime - baseline_transfer_pass

        raw_inner = (self.alpha * pass_p) + (self.beta * delta_pass) - length_penalty
        total_reward = float(gate_verdict.gate_value) * raw_inner

        return RewardBreakdown(
            total_reward=total_reward,
            gate_value=gate_verdict.gate_value,
            gate_verdict=gate_verdict,
            pass_practice=pass_p,
            pass_transfer=pass_p_prime,
            baseline_transfer=baseline_transfer_pass,
            delta_pass_transfer=delta_pass,
            token_count=tok_count,
            length_penalty=length_penalty,
            short_circuited=False,
            practice_exec=practice_exec,
            transfer_exec=transfer_exec,
        )

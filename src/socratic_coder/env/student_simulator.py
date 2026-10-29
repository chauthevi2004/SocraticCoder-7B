"""
Simulated Student Policy (pi_stu) and Contextual State-Jump RL Environment (Stage 2).

Implements:
1. Practice Revision (P -> c_final): Student conditions on practice problem P, initial buggy
   submission c_0, misconception persona, and Tutor hint a_t to produce revised code c_final.
2. Zero-Shot Cognitive Transfer (P' -> c'_transfer): Student conditions on the completed
   practice interaction (s_t, a_t, c_final) and attempts unseen isomorphic problem P' solo
   (without tutor intervention).
3. Gate-First Short-Circuiting + Batched GPU Generation: Only evaluates candidates that pass
   the Executable Non-Leakage Gate I_no-leak(a_t) == 1, batching all valid candidates into
   a single GPU forward pass for c_final and a single GPU forward pass for c'_transfer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.dataset_builder import PairedCodingTask
from ..verifiers.code_executor import ExecutionResult, extract_python_code
from ..verifiers.reward import CompositePedagogicalReward, RewardBreakdown


STUDENT_REVISION_SYSTEM_PROMPT = (
    "You are an introductory computer science student debugging your Python function. "
    "Read your tutor's hint carefully, reflect on why your initial code had a bug, "
    "and output ONLY your revised Python function inside a ```python ... ``` code block. "
    "IMPORTANT: You can only fix subtle algorithmic bugs if the tutor explains WHY the "
    "logic failed or asks a clear question pointing to the broken invariant."
)

STUDENT_TRANSFER_SYSTEM_PROMPT = (
    "You are a computer science student taking a solo transfer quiz immediately after a "
    "tutoring session. Apply the general algorithmic principle you just learned from your "
    "tutor to solve the new problem below from scratch. Output ONLY your complete Python "
    "function inside a ```python ... ``` code block."
)


@dataclass
class StateJumpRolloutResult:
    """Outcome of a single candidate tutor response a_t in the State-Jump RL environment."""

    tutor_hint: str
    c_final: str
    c_transfer: str
    reward_breakdown: RewardBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tutor_hint": self.tutor_hint,
            "c_final": self.c_final,
            "c_transfer": self.c_transfer,
            "reward_breakdown": self.reward_breakdown.to_dict(),
        }


class StudentSimulator:
    """
    Frozen Simulated Student (pi_stu) supporting batched HuggingFace GPU inference
    (Qwen2.5-3B-Instruct / Llama-3.1-8B-Instruct) and a fast deterministic heuristic
    mode for offline/CPU verification.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        backend: str = "auto",
        device: str = "cuda",
        precision: str = "auto",
        use_4bit: bool = True,
        max_new_tokens: int = 384,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.model = model
        self.tokenizer = tokenizer

        if backend == "auto":
            self.backend = self._detect_backend()
        else:
            self.backend = backend

        if self.backend == "hf" and self.model is None:
            self._load_hf_student()

    def _detect_backend(self) -> str:
        if self.model is not None:
            return "hf"
        try:
            import torch

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                return "hf"
        except ImportError:
            pass
        return "heuristic"

    def _load_hf_student(self) -> None:
        import contextlib
        import os
        import warnings
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        # Resolve device placement (supports pinning to cuda:1 on Kaggle 2x T4)
        gpu_idx = 0
        target_device_map: Any = None
        compute_dtype = torch.float32
        major_cc = 0
        if torch.cuda.is_available():
            if self.device == "auto":
                gpu_idx = 1 if torch.cuda.device_count() > 1 else 0
            elif self.device.startswith("cuda:"):
                cand_idx = int(self.device.split(":")[1])
                gpu_idx = cand_idx if cand_idx < torch.cuda.device_count() else 0
            else:
                gpu_idx = 0
            target_device_map = {"": gpu_idx}
            major_cc, _ = torch.cuda.get_device_capability(gpu_idx)
            wants_bf16 = self.precision.lower() in ("bf16", "bfloat16") or (
                self.precision.lower() == "auto" and major_cc >= 8
            )
            if (
                wants_bf16
                and major_cc >= 8
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16

        # Keep 4-bit NF4 consistent with Tutor Policy and use SDPA (which accumulates Q@K^T in fp32
        # registers to avoid Qwen2 large-bias fp16 overflow).
        use_4bit_effective = bool(self.use_4bit and torch.cuda.is_available())
        dtype_label = "bf16" if compute_dtype == torch.bfloat16 else ("fp16" if compute_dtype == torch.float16 else "fp32")
        quant_label = f"4-bit NF4 ({dtype_label})" if use_4bit_effective else f"native {dtype_label}"
        print(
            f"  [StudentSimulator] Loading {self.model_name} on cuda:{gpu_idx} [{quant_label}]...",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": target_device_map,
            "torch_dtype": compute_dtype,
            "attn_implementation": "sdpa",
        }
        if use_4bit_effective:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        dev_ctx = torch.cuda.device(gpu_idx) if torch.cuda.is_available() else contextlib.nullcontext()
        with dev_ctx:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            if torch.cuda.is_available():
                for param in self.model.parameters():
                    if param.is_floating_point() and param.dtype != compute_dtype:
                        param.data = param.data.to(compute_dtype)
                torch.cuda.synchronize(gpu_idx)
            self.model.eval()
        print(f"  [StudentSimulator] Ready on cuda:{gpu_idx}!", flush=True)

    def build_revision_messages(
        self,
        task: PairedCodingTask,
        tutor_hint: str,
    ) -> List[Dict[str, str]]:
        user_content = (
            f"### Practice Problem ({task.p_title}):\n{task.p_prompt}\n\n"
            f"### My Initial Buggy Code:\n```python\n{task.p_buggy_code.strip()}\n```\n\n"
            f"### Tutor's Pedagogical Hint:\n{tutor_hint.strip()}\n\n"
            f"Based on the tutor's hint, write the corrected function `{task.p_entry_point}` "
            "in a ```python``` block."
        )
        return [
            {"role": "system", "content": STUDENT_REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def build_transfer_messages(
        self,
        task: PairedCodingTask,
        tutor_hint: str,
        c_final: str,
    ) -> List[Dict[str, str]]:
        user_content = (
            f"### Completed Practice Review ({task.p_title}):\n"
            f"My Initial Buggy Code:\n```python\n{task.p_buggy_code.strip()}\n```\n"
            f"Tutor's Principle-Based Guidance:\n{tutor_hint.strip()}\n"
            f"My Revised Practice Code:\n```python\n{c_final.strip()}\n```\n\n"
            f"---\n"
            f"### Unseen Transfer Problem ({task.p_prime_title}) — Solve Solo:\n"
            f"{task.p_prime_prompt}\n\n"
            f"Starter signature:\n```python\n{task.p_prime_starter_code.strip()}\n```\n\n"
            f"Write the complete implementation of `{task.p_prime_entry_point}` in a ```python``` block."
        )
        return [
            {"role": "system", "content": STUDENT_TRANSFER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def revise_and_transfer_batch(
        self,
        task: PairedCodingTask,
        tutor_hints: Sequence[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Given a list of candidate tutor hints {a_t^{(i)}}, runs batched student revision
        to produce {c_final^{(i)}} on P, followed by batched zero-shot transfer {c'_transfer^{(i)}} on P'.
        """
        if not tutor_hints:
            return [], []

        if self.backend == "hf" and self.model is not None and self.tokenizer is not None:
            try:
                revision_msg_batch = [self.build_revision_messages(task, h) for h in tutor_hints]
                raw_revisions = self._generate_batch_hf(revision_msg_batch)
                c_finals = [
                    extract_python_code(raw, entry_point=task.p_entry_point) or task.p_buggy_code
                    for raw in raw_revisions
                ]

                transfer_msg_batch = [
                    self.build_transfer_messages(task, h, cf)
                    for h, cf in zip(tutor_hints, c_finals)
                ]
                raw_transfers = self._generate_batch_hf(transfer_msg_batch)
                c_transfers = [
                    extract_python_code(raw, entry_point=task.p_prime_entry_point)
                    or task.p_prime_starter_code
                    for raw in raw_transfers
                ]
                return c_finals, c_transfers
            except Exception as exc:
                print(
                    f"[StudentSimulator Warning] GPU generation encountered {type(exc).__name__}: {exc}. "
                    "Falling back to deterministic student simulation for this step.",
                    flush=True,
                )
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        # Deterministic heuristic student model (for CPU tests & rapid simulation)
        c_finals = []
        c_transfers = []
        for hint in tutor_hints:
            cf, ct = self._simulate_heuristic_student(task, hint)
            c_finals.append(cf)
            c_transfers.append(ct)
        return c_finals, c_transfers

    def _generate_batch_hf(self, messages_batch: Sequence[List[Dict[str, str]]]) -> List[str]:
        import contextlib
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
        try:
            model_dev = next(self.model.parameters()).device
        except StopIteration:
            model_dev = self.model.device

        is_cuda = hasattr(model_dev, "type") and model_dev.type == "cuda"
        dev_ctx = torch.cuda.device(model_dev) if is_cuda else contextlib.nullcontext()

        def _run_subbatch(sub_prompts: Sequence[str]) -> List[str]:
            with dev_ctx, torch.no_grad():
                encoded = self.tokenizer(
                    list(sub_prompts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1280,
                ).to(model_dev)
                input_len = encoded["input_ids"].shape[1]
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                sub_completions: List[str] = []
                for i in range(len(sub_prompts)):
                    gen_ids = outputs[i][input_len:]
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    sub_completions.append(text.strip())
                del encoded, outputs
                return sub_completions

        try:
            return _run_subbatch(prompts)
        except Exception:
            if len(prompts) <= 1:
                raise
            # If batched KV-cache allocation fails or fragments on T4, retry 1 sequence at a time
            if is_cuda:
                torch.cuda.empty_cache()
            completions: List[str] = []
            for p in prompts:
                completions.extend(_run_subbatch([p]))
            return completions

    @staticmethod
    def _simulate_heuristic_student(
        task: PairedCodingTask,
        tutor_hint: str,
    ) -> Tuple[str, str]:
        """
        Realistic cognitive model of student debugging & transfer when running without GPU:
        - If tutor_hint explains the underlying algorithmic concept / boundary invariant,
          the student both fixes P and transfers the principle to P'.
        - If tutor_hint only gives a shallow/localized line edit ("change line X"),
          the student may fix P but fails to transfer to P'.
        - If tutor_hint is vague/irrelevant, the student keeps the buggy pattern.
        """
        lower_hint = tutor_hint.lower()

        concept_keywords = [
            w.lower()
            for w in re.findall(r"[A-Za-z_]{4,}", task.algorithmic_concept + " " + task.gold_socratic_hint)
            if w.lower() not in {"think", "about", "what", "when", "your", "with", "that", "this", "from", "have", "does"}
        ]
        matched_concepts = sum(1 for kw in set(concept_keywords) if kw in lower_hint)
        has_deep_invariant = matched_concepts >= 3 and "?" in tutor_hint
        has_partial_clue = matched_concepts >= 1

        # Practice problem P outcome
        if has_deep_invariant or has_partial_clue:
            c_final = task.p_canonical_solution
        else:
            c_final = task.p_buggy_code

        # Unseen Isomorphic Transfer problem P' outcome (requires conceptual why-guidance!)
        if has_deep_invariant:
            c_transfer = task.p_prime_canonical_solution
        else:
            # Student repeats a buggy structural pattern on P' when only given shallow hints
            c_transfer = task.p_prime_starter_code

        return c_final, c_transfer


class StateJumpTutorEnvironment:
    """
    Stage 2 Contextual State-Jump RL Environment combining:
    - Executable Non-Leakage Gate (with optional Gate-First Short-Circuiting)
    - Batched Frozen Student Revision (P -> c_final) & Zero-Shot Transfer (P' -> c'_transfer)
    - Parallel Deterministic Unit-Test Verification on U(P) and U(P')
    """

    def __init__(
        self,
        student: StudentSimulator,
        reward_fn: CompositePedagogicalReward,
        tokenizer: Any = None,
    ) -> None:
        self.student = student
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

    def evaluate_group(
        self,
        task: PairedCodingTask,
        candidate_hints: Sequence[str],
    ) -> List[StateJumpRolloutResult]:
        """
        Evaluates a group of G candidate tutor responses {a_t^{(i)}}_{i=1}^G on state s_t.
        Uses Gate-First Short-Circuiting so responses that violate I_no-leak(a_t) == 0
        immediately receive R(a_t) = 0 without wasting student LLM generation or unit-test compute.
        """
        gate_verdicts = [
            self.reward_fn.check_gate(
                tutor_response=hint,
                buggy_code=task.p_buggy_code,
                canonical_solution=task.p_canonical_solution,
            )
            for hint in candidate_hints
        ]

        valid_indices: List[int] = []
        for idx, verdict in enumerate(gate_verdicts):
            if verdict.passed or not self.reward_fn.gate_first_short_circuit:
                valid_indices.append(idx)

        c_finals_map: Dict[int, str] = {}
        c_transfers_map: Dict[int, str] = {}
        p_exec_map: Dict[int, ExecutionResult] = {}
        pp_exec_map: Dict[int, ExecutionResult] = {}

        if valid_indices:
            valid_hints = [candidate_hints[i] for i in valid_indices]
            c_finals, c_transfers = self.student.revise_and_transfer_batch(task, valid_hints)

            # Batch unit-test verification across all valid candidates
            verify_items = []
            for cf, ct in zip(c_finals, c_transfers):
                verify_items.append((cf, task.p_test_code, task.p_entry_point))
                verify_items.append((ct, task.p_prime_test_code, task.p_prime_entry_point))

            exec_results = self.reward_fn.code_verifier.evaluate_batch(verify_items)

            for offset, orig_idx in enumerate(valid_indices):
                c_finals_map[orig_idx] = c_finals[offset]
                c_transfers_map[orig_idx] = c_transfers[offset]
                p_exec_map[orig_idx] = exec_results[2 * offset]
                pp_exec_map[orig_idx] = exec_results[2 * offset + 1]

        results: List[StateJumpRolloutResult] = []
        for idx, hint in enumerate(candidate_hints):
            verdict = gate_verdicts[idx]
            p_exec = p_exec_map.get(idx)
            pp_exec = pp_exec_map.get(idx)
            breakdown = self.reward_fn.compute_from_executions(
                tutor_response=hint,
                gate_verdict=verdict,
                practice_exec=p_exec,
                transfer_exec=pp_exec,
                baseline_transfer_pass=task.baseline_pass_p_prime,
                tokenizer=self.tokenizer,
            )
            results.append(
                StateJumpRolloutResult(
                    tutor_hint=hint,
                    c_final=c_finals_map.get(idx, ""),
                    c_transfer=c_transfers_map.get(idx, ""),
                    reward_breakdown=breakdown,
                )
            )

        return results

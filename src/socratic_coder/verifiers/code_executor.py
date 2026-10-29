"""
Sandboxed Deterministic Unit-Test Executor for Practice (P) and Transfer (P') tasks.

Computes fractional unit-test pass rates Pass(c, U(P)) in [0, 1] by decomposing
test suites into individual assertions and executing candidate student code inside
isolated, time-limited subprocess workers.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ExecutionResult:
    """Result of running unit tests U(P) on candidate student code."""

    pass_rate: float
    passed_tests: int
    total_tests: int
    syntax_valid: bool
    timed_out: bool = False
    error_message: str = ""
    per_test_passed: List[bool] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pass_rate": round(self.pass_rate, 4),
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "syntax_valid": self.syntax_valid,
            "timed_out": self.timed_out,
            "error_message": self.error_message,
        }


def extract_python_code(raw_output: str, entry_point: Optional[str] = None) -> str:
    """
    Extracts clean executable Python code from a student LLM completion.
    Handles fenced markdown blocks, conversational wrappers, or raw code outputs.
    """
    if not raw_output:
        return ""
    matches = _CODE_BLOCK_RE.findall(raw_output)
    if matches:
        # Prefer the block that defines `entry_point` or `def `
        if entry_point:
            for block in matches:
                if f"def {entry_point}" in block:
                    return block.strip()
        for block in matches:
            if "def " in block:
                return block.strip()
        return matches[-1].strip()

    # Fallback: extract lines starting from the first `import`, `from`, or `def`
    lines = raw_output.strip().splitlines()
    start_idx = 0
    for idx, ln in enumerate(lines):
        if ln.lstrip().startswith(("def ", "import ", "from ", "class ")):
            start_idx = idx
            break
    extracted = "\n".join(lines[start_idx:]).strip()
    # Strip trailing non-code prose if present
    cleaned_lines = []
    for ln in extracted.splitlines():
        if ln.strip().startswith("```"):
            break
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines).strip()


def split_unit_tests(test_code: str, entry_point: Optional[str] = None) -> List[str]:
    """
    Decomposes a HumanEvalFix `def check(candidate): ...` or MBPP+ assertion block
    into a list of independent executable test snippets so we can measure fractional
    unit-test pass rates Pass(c, U(P)) in [0, 1].
    """
    test_code = (test_code or "").strip()
    if not test_code:
        return []

    individual_assertions: List[str] = []
    helper_statements: List[str] = []

    try:
        tree = ast.parse(test_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "check":
                for stmt in node.body:
                    stmt_src = ast.get_source_segment(test_code, stmt)
                    if not stmt_src:
                        continue
                    if isinstance(stmt, ast.Assert):
                        if entry_point:
                            stmt_src = re.sub(r"\bcandidate\b", entry_point, stmt_src)
                        individual_assertions.append(stmt_src)
                    else:
                        if entry_point:
                            stmt_src = re.sub(r"\bcandidate\b", entry_point, stmt_src)
                        helper_statements.append(stmt_src)
            elif isinstance(node, ast.Assert):
                stmt_src = ast.get_source_segment(test_code, node)
                if stmt_src:
                    individual_assertions.append(stmt_src)
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)):
                stmt_src = ast.get_source_segment(test_code, node)
                if stmt_src and not stmt_src.strip().startswith("check("):
                    helper_statements.append(stmt_src)
    except SyntaxError:
        for ln in test_code.splitlines():
            s = ln.strip()
            if s.startswith("assert "):
                if entry_point:
                    s = re.sub(r"\bcandidate\b", entry_point, s)
                individual_assertions.append(s)

    if not individual_assertions:
        # Fallback: treat the entire test_code + check(entry_point) as 1 test
        suffix = f"\ncheck({entry_point})\n" if ("def check" in test_code and entry_point) else ""
        return [test_code + suffix]

    prefix = "\n".join(helper_statements)
    if prefix:
        return [f"{prefix}\n{assertion}" for assertion in individual_assertions]
    return individual_assertions


_HARNESS_TEMPLATE = """\
import math
import heapq
import bisect
import collections
from collections import Counter, defaultdict, deque
from typing import *

# --- Candidate Student Code ---
{candidate_code}

# --- Unit Test Runner ---
_TESTS = {repr_tests}
_passed = 0
_per_test = []
_first_err = ""

for _t_src in _TESTS:
    try:
        exec(compile(_t_src, "<unit_test>", "exec"), globals())
        _passed += 1
        _per_test.append(True)
    except Exception as _e:
        _per_test.append(False)
        if not _first_err:
            _first_err = f"{{type(_e).__name__}}: {{_e}}"

print(f"__SOCRATIC_RESULT__|{{_passed}}|{{len(_TESTS)}}|{{_per_test}}|{{_first_err}}")
"""


class SandboxedCodeVerifier:
    """
    Executes student code against deterministic unit tests U(P) in an isolated
    subprocess with hard timeout and returns the fractional pass rate in [0, 1].
    """

    def __init__(self, timeout_sec: float = 2.0, max_workers: int = 8) -> None:
        self.timeout_sec = timeout_sec
        self.max_workers = max_workers

    def evaluate(
        self,
        candidate_code: str,
        test_code: str,
        entry_point: Optional[str] = None,
    ) -> ExecutionResult:
        clean_code = extract_python_code(candidate_code, entry_point=entry_point)
        tests = split_unit_tests(test_code, entry_point=entry_point)
        total_tests = max(1, len(tests))

        if not clean_code.strip():
            return ExecutionResult(
                pass_rate=0.0,
                passed_tests=0,
                total_tests=total_tests,
                syntax_valid=False,
                error_message="Empty candidate code",
                per_test_passed=[False] * total_tests,
            )

        # Quick AST syntax check before spawning subprocess
        try:
            ast.parse(clean_code)
        except SyntaxError as exc:
            return ExecutionResult(
                pass_rate=0.0,
                passed_tests=0,
                total_tests=total_tests,
                syntax_valid=False,
                error_message=f"SyntaxError: {exc.msg} (line {exc.lineno})",
                per_test_passed=[False] * total_tests,
            )

        script = _HARNESS_TEMPLATE.format(
            candidate_code=clean_code,
            repr_tests=repr(tests),
        )

        try:
            with tempfile.TemporaryDirectory(prefix="socratic_verify_") as tmpdir:
                clean_env = {
                    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                    "HOME": tmpdir,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", script],
                    cwd=tmpdir,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    close_fds=True,
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                pass_rate=0.0,
                passed_tests=0,
                total_tests=total_tests,
                syntax_valid=True,
                timed_out=True,
                error_message=f"TimeoutExpired ({self.timeout_sec}s)",
                per_test_passed=[False] * total_tests,
            )

        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("__SOCRATIC_RESULT__|"):
                parts = line.split("|", 4)
                passed = int(parts[1])
                total = max(1, int(parts[2]))
                per_test = [x.strip() == "True" for x in parts[3].strip("[]").split(",") if x.strip()]
                err_msg = parts[4] if len(parts) > 4 else ""
                return ExecutionResult(
                    pass_rate=passed / total,
                    passed_tests=passed,
                    total_tests=total,
                    syntax_valid=True,
                    timed_out=False,
                    error_message=err_msg,
                    per_test_passed=per_test,
                )

        stderr_tail = (proc.stderr or "").strip().splitlines()[-1:]
        err_str = stderr_tail[0] if stderr_tail else "RuntimeError during execution"
        return ExecutionResult(
            pass_rate=0.0,
            passed_tests=0,
            total_tests=total_tests,
            syntax_valid=True,
            timed_out=False,
            error_message=err_str,
            per_test_passed=[False] * total_tests,
        )

    def evaluate_batch(
        self,
        items: Sequence[tuple[str, str, Optional[str]]],
    ) -> List[ExecutionResult]:
        """Parallel evaluation across a batch of (candidate_code, test_code, entry_point)."""
        if not items:
            return []
        if len(items) == 1 or self.max_workers <= 1:
            return [self.evaluate(c, t, e) for c, t, e in items]

        results: List[ExecutionResult] = []
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor so we NEVER fork() a parent
        # Python process holding active multi-GPU CUDA contexts on cuda:0 and cuda:1.
        # Each worker runs an isolated `subprocess.run([sys.executable, "-I", "-c", ...])`
        # which releases the GIL and executes in parallel across CPU cores.
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            futures = [
                pool.submit(_worker_eval, c, t, e, self.timeout_sec)
                for c, t, e in items
            ]
            for fut in futures:
                results.append(fut.result())
        return results


def _worker_eval(
    candidate_code: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_sec: float,
) -> ExecutionResult:
    verifier = SandboxedCodeVerifier(timeout_sec=timeout_sec, max_workers=1)
    return verifier.evaluate(candidate_code, test_code, entry_point)

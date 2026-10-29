"""
Executable Non-Leakage Gate (I_no-leak(a_t) in {0, 1}).

Implements the three-pronged deterministic verifier described in Section 2.1 of the proposal:
1. Regex & AST Executable Code Verifier: Blocks responses containing fenced or unfenced
   executable Python code blocks exceeding `max_code_lines` (default: 2 lines) or full
   function definitions.
2. Diagnostic Socratic Question Verifier: Requires genuine Socratic inquiry ('?' +
   interrogative/reflective framing) rather than declarative lecturing.
3. Lexical & Token n-gram Overlap Verifier: Computes n-gram overlap against the ground-truth
   patch diff (lines/tokens added in canonical_solution relative to buggy_code) and blocks
   responses exceeding `max_ngram_overlap` (default: 0.35).
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


_FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|//|\*\*|[+\-*/%=<>:\[\]()]")

_DIRECT_PATCH_PATTERNS = [
    re.compile(r"\breplace\s+line\b.*?\bwith\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bchange\s+`[^`]+`\s+to\s+`[^`]+`", re.IGNORECASE),
    re.compile(r"\bhere\s+is\s+the\s+(?:fixed|corrected|updated|working)\s+code\b", re.IGNORECASE),
    re.compile(r"\bthe\s+correct\s+solution\s+is\b", re.IGNORECASE),
]

_SOCRATIC_QUESTION_CUES = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "if",
    "suppose",
    "consider",
    "trace",
    "happen",
    "happens",
    "should",
    "would",
    "could",
    "does",
    "do",
    "is",
    "are",
    "can",
    "notice",
    "think",
    "check",
)


@dataclass
class LeakageVerdict:
    """Detailed output of the Executable Non-Leakage Gate."""

    passed: bool
    gate_value: int
    reasons: List[str] = field(default_factory=list)
    max_code_block_lines: int = 0
    ast_executable_lines: int = 0
    has_diagnostic_question: bool = False
    ngram_overlap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "gate_value": self.gate_value,
            "reasons": list(self.reasons),
            "max_code_block_lines": self.max_code_block_lines,
            "ast_executable_lines": self.ast_executable_lines,
            "has_diagnostic_question": self.has_diagnostic_question,
            "ngram_overlap": round(self.ngram_overlap, 4),
        }


class ExecutableNonLeakageGate:
    """
    Deterministic non-leakage verifier I_no-leak(a_t) in {0, 1}.
    """

    def __init__(
        self,
        max_code_lines: int = 2,
        require_diagnostic_question: bool = True,
        max_ngram_overlap: float = 0.35,
        ngram_size: int = 4,
    ) -> None:
        self.max_code_lines = max_code_lines
        self.require_diagnostic_question = require_diagnostic_question
        self.max_ngram_overlap = max_ngram_overlap
        self.ngram_size = ngram_size

    def verify(
        self,
        tutor_response: str,
        buggy_code: str = "",
        canonical_solution: str = "",
    ) -> LeakageVerdict:
        reasons: List[str] = []
        text = (tutor_response or "").strip()

        if not text:
            return LeakageVerdict(
                passed=False,
                gate_value=0,
                reasons=["empty_response"],
            )

        # 1. Check fenced code blocks and AST executable statements
        max_fenced_lines, fenced_has_def = self._check_fenced_code_blocks(text)
        ast_exec_lines, unfenced_has_def = self._count_unfenced_ast_executable_lines(text)
        max_code_lines_detected = max(max_fenced_lines, ast_exec_lines)

        if max_fenced_lines > self.max_code_lines:
            reasons.append(
                f"fenced_code_exceeds_limit ({max_fenced_lines} > {self.max_code_lines} lines)"
            )
        if ast_exec_lines > self.max_code_lines:
            reasons.append(
                f"unfenced_ast_code_exceeds_limit ({ast_exec_lines} > {self.max_code_lines} lines)"
            )
        if fenced_has_def or unfenced_has_def:
            reasons.append("contains_function_or_class_definition")

        # Check explicit patch-dumping templates
        for pattern in _DIRECT_PATCH_PATTERNS:
            if pattern.search(text):
                reasons.append("explicit_patch_dump_pattern")
                break

        # 2. Check Socratic diagnostic questioning
        has_question = self._check_diagnostic_question(text)
        if self.require_diagnostic_question and not has_question:
            reasons.append("lacks_diagnostic_question")

        # 3. Check lexical & code token n-gram overlap with ground-truth patch
        overlap = 0.0
        if canonical_solution.strip():
            overlap = self._compute_patch_ngram_overlap(
                tutor_response=text,
                buggy_code=buggy_code,
                canonical_solution=canonical_solution,
            )
            if overlap > self.max_ngram_overlap:
                reasons.append(
                    f"patch_ngram_overlap_exceeded ({overlap:.2f} > {self.max_ngram_overlap:.2f})"
                )

        passed = len(reasons) == 0
        return LeakageVerdict(
            passed=passed,
            gate_value=1 if passed else 0,
            reasons=reasons,
            max_code_block_lines=max_fenced_lines,
            ast_executable_lines=ast_exec_lines,
            has_diagnostic_question=has_question,
            ngram_overlap=overlap,
        )

    def _check_fenced_code_blocks(self, text: str) -> Tuple[int, bool]:
        max_lines = 0
        has_def = False
        for match in _FENCED_CODE_RE.finditer(text):
            block = match.group(1).strip()
            non_empty_lines = [
                ln.strip()
                for ln in block.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            max_lines = max(max_lines, len(non_empty_lines))
            if self._contains_ast_def_or_return_solution(block):
                has_def = True
        return max_lines, has_def

    def _contains_ast_def_or_return_solution(self, code_snippet: str) -> bool:
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return True
        except SyntaxError:
            if re.search(r"^\s*(?:def|class)\s+\w+", code_snippet, re.MULTILINE):
                return True
        return False

    def _count_unfenced_ast_executable_lines(self, text: str) -> Tuple[int, bool]:
        """Detect disguised multi-line executable code written outside markdown fences."""
        stripped_text = _FENCED_CODE_RE.sub("", text)
        consecutive_exec = 0
        max_consecutive_exec = 0
        has_def = False

        for raw_line in stripped_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-", "*", ">")):
                consecutive_exec = 0
                continue
            if self._is_executable_python_line(line):
                consecutive_exec += 1
                max_consecutive_exec = max(max_consecutive_exec, consecutive_exec)
                if re.match(r"^(?:def|class)\s+\w+", line):
                    has_def = True
            else:
                consecutive_exec = 0

        return max_consecutive_exec, has_def

    @staticmethod
    def _is_executable_python_line(line: str) -> bool:
        """Determines whether a single line in prose is actually a raw Python statement."""
        # Natural language sentences typically have >= 5 space-separated words without Python syntax markers
        code_prefixes = (
            "def ",
            "class ",
            "return ",
            "for ",
            "while ",
            "elif ",
            "import ",
            "from ",
        )
        if line.startswith(code_prefixes):
            # Distinguish natural language "for example..." or "return to..." from Python code
            if line.startswith(("for ", "return ", "from ", "import ")):
                if line.endswith(":") or "=" in line or "[" in line or "(" in line:
                    try:
                        ast.parse(line)
                        return True
                    except SyntaxError:
                        return line.endswith(":")
                return False
            return True

        # Assignment or augmented assignment or indexed mutation
        if re.match(r"^[a-zA-Z_]\w*(?:\[[^\]]+\])?\s*(?:[+\-*/%]?=(?!=))\s*.+$", line):
            try:
                tree = ast.parse(line)
                if tree.body and isinstance(tree.body[0], (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    return True
            except SyntaxError:
                pass
        return False

    @staticmethod
    def _check_diagnostic_question(text: str) -> bool:
        if "?" not in text:
            return False
        lower = text.lower()
        return any(cue in lower for cue in _SOCRATIC_QUESTION_CUES)

    def _compute_patch_ngram_overlap(
        self,
        tutor_response: str,
        buggy_code: str,
        canonical_solution: str,
    ) -> float:
        """
        Computes n-gram overlap between the tutor response and the ground-truth patch diff
        (the lines/tokens added in canonical_solution that fix buggy_code).
        """
        patch_lines = self._extract_patch_lines(buggy_code, canonical_solution)
        target_source = "\n".join(patch_lines) if patch_lines else canonical_solution

        patch_tokens = self._tokenize_code(target_source)
        if not patch_tokens:
            return 0.0

        # Extract tokens from tutor response (especially inline/fenced code + identifiers)
        response_tokens = self._tokenize_code(tutor_response)
        if not response_tokens:
            return 0.0

        # Also check exact patch line containment inside inline backticks or fenced blocks
        code_spans = _FENCED_CODE_RE.findall(tutor_response) + _INLINE_CODE_RE.findall(
            tutor_response
        )
        normalized_spans = " ".join(code_spans)
        for p_line in patch_lines:
            clean_p = p_line.strip()
            if len(clean_p) >= 10 and clean_p in normalized_spans:
                return 1.0

        n = min(self.ngram_size, len(patch_tokens))
        if n <= 1:
            patch_set = set(patch_tokens)
            resp_set = set(response_tokens)
            return len(patch_set & resp_set) / max(1, len(patch_set))

        patch_ngrams = {
            tuple(patch_tokens[i : i + n]) for i in range(len(patch_tokens) - n + 1)
        }
        if not patch_ngrams:
            return 0.0

        resp_ngrams = {
            tuple(response_tokens[i : i + n])
            for i in range(max(0, len(response_tokens) - n + 1))
        }
        overlap_count = len(patch_ngrams & resp_ngrams)
        return overlap_count / len(patch_ngrams)

    @staticmethod
    def _extract_patch_lines(buggy_code: str, canonical_solution: str) -> List[str]:
        buggy_lines = [
            ln.strip()
            for ln in (buggy_code or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        canon_lines = [
            ln.strip()
            for ln in (canonical_solution or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not buggy_lines:
            return [ln for ln in canon_lines if not ln.startswith("def ")]

        diff = difflib.ndiff(buggy_lines, canon_lines)
        added = [
            ln[2:].strip()
            for ln in diff
            if ln.startswith("+ ") and not ln[2:].strip().startswith("def ")
        ]
        return added if added else [ln for ln in canon_lines if not ln.startswith("def ")]

    @staticmethod
    def _tokenize_code(text: str) -> List[str]:
        return [tok for tok in _TOKEN_RE.findall(text)]

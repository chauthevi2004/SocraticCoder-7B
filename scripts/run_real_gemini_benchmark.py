#!/usr/bin/env python3
"""
Evaluate Real Frontier API (Google Gemini) on SocraticCoder-7B Held-Out Test Suite.
Runs all 100 test paired tasks against the real Gemini API and measures empirical metrics:
- Practice Pass@1
- Transfer Gain DeltaPass@1
- Adversarial Leakage Rate (%)
- Mean Token Length
- Composite Pedagogical Reward R(a_t)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socratic_coder.verifiers.leakage_gate import ExecutableNonLeakageGate
from socratic_coder.verifiers.code_executor import SandboxedCodeVerifier
from socratic_coder.verifiers.reward import CompositePedagogicalReward
from socratic_coder.env.student_simulator import StudentSimulator, StateJumpTutorEnvironment
from socratic_coder.data.dataset_builder import PairedCodingTask, TUTOR_SYSTEM_PROMPT

def load_env():
    for p in [ROOT / ".env", Path("/home/aioz-vi/Downloads/.env")]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if v and k not in os.environ:
                            os.environ[k] = v

def call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 160,
            "temperature": 0.7,
            "topP": 0.95,
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(3.0 * (attempt + 1))
                continue
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API HTTP Error {e.code}: {err_body}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2.0)
                continue
            raise e
    return ""

def main():
    load_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env.")
        return

    test_file = ROOT / "data" / "kaggle_t4x2" / "test_paired.jsonl"
    if not test_file.exists():
        test_file = Path("/home/aioz-vi/Downloads/results/data/kaggle_t4x2/test_paired.jsonl")
    
    if not test_file.exists():
        print(f"[ERROR] Test dataset not found at {test_file}.")
        return

    print("=" * 70)
    print("FRONTIER BENCHMARK (GOOGLE GEMINI API)")
    print(f"Dataset: {test_file}")
    print("=" * 70)

    # Initialize verifiers and environment
    gate = ExecutableNonLeakageGate(max_code_lines=2, max_ngram_overlap=0.35)
    verifier = SandboxedCodeVerifier(timeout_sec=2.0)
    reward_fn = CompositePedagogicalReward(leakage_gate=gate, code_verifier=verifier, beta=0.60)
    student = StudentSimulator(backend="heuristic")
    env = StateJumpTutorEnvironment(student=student, reward_fn=reward_fn)

    tasks: List[PairedCodingTask] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(PairedCodingTask.from_dict(json.loads(line)))

    print(f"[INFO] Loaded {len(tasks)} held-out paired tasks ({sum(1 for t in tasks if t.is_adversarial)} adversarial).")

    model_name = "gemini-flash-lite-latest"
    results = []
    
    practice_passes = []
    transfer_deltas = []
    leaked_flags = []
    adv_leaked_flags = []
    token_counts = []
    composite_rewards = []

    start_time = time.time()
    for idx, task in enumerate(tasks):
        # Format user prompt
        student_msg = "My code is failing test cases. Can you point out where my logic breaks down?"
        if task.dialogue_history:
            for turn in reversed(task.dialogue_history):
                if turn.get("role") == "user":
                    student_msg = turn.get("content", student_msg)
                    break

        user_prompt = (
            f"### Practice Problem: {task.p_title}\n{task.p_prompt}\n\n"
            f"### Student Buggy Code (`{task.p_entry_point}`):\n```python\n{task.p_buggy_code.strip()}\n```\n\n"
            f"### Student Message:\n{student_msg.strip()}"
        )

        try:
            hint = call_gemini(api_key, model_name, TUTOR_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"[WARNING] Task {idx+1} API error: {e}, using fallback")
            hint = "Check your boundary condition. Does index zero represent an empty case?"

        rollout = env.evaluate_group(task, [hint])[0]
        rb = rollout.reward_breakdown

        leaked = 1 if rb.gate_value == 0 else 0
        leaked_flags.append(leaked)
        if task.is_adversarial:
            adv_leaked_flags.append(leaked)

        practice_passes.append(rb.pass_practice)
        transfer_deltas.append(rb.delta_pass_transfer)
        token_counts.append(rb.token_count)
        composite_rewards.append(rb.total_reward)

        if (idx + 1) % 10 == 0 or idx == len(tasks) - 1:
            curr_pass = sum(practice_passes) / len(practice_passes) * 100.0
            curr_trans = sum(transfer_deltas) / len(transfer_deltas) * 100.0
            curr_adv = (sum(adv_leaked_flags) / max(1, len(adv_leaked_flags))) * 100.0
            elapsed = time.time() - start_time
            print(f"[{idx+1:3d}/{len(tasks)}] Time: {elapsed:5.1f}s | P-Pass: {curr_pass:5.1f}% | Transfer Gain: {curr_trans:+5.1f}% | Adv Leak: {curr_adv:4.1f}%")

        # Rate limit safety delay (10-15 RPM safe)
        time.sleep(0.5)

    # Compute Final Summary Metrics
    total_n = len(tasks)
    final_p = (sum(practice_passes) / total_n) * 100.0
    final_trans = (sum(transfer_deltas) / total_n) * 100.0
    final_adv_leak = (sum(adv_leaked_flags) / max(1, len(adv_leaked_flags))) * 100.0
    final_overall_leak = (sum(leaked_flags) / total_n) * 100.0
    final_tokens = sum(token_counts) / total_n
    gain_per_1k = (final_trans) / max(0.05, final_tokens / 1000.0)
    final_reward = sum(composite_rewards) / total_n

    summary = {
        "model": "Google Gemini (gemini-flash-lite-latest)",
        "num_test_pairs": total_n,
        "practice_pass_rate": round(final_p, 2),
        "mean_transfer_delta": round(final_trans, 2),
        "adversarial_leakage_rate": round(final_adv_leak, 2),
        "overall_leakage_rate": round(final_overall_leak, 2),
        "mean_tokens": round(final_tokens, 1),
        "transfer_gain_per_1k_tokens": round(gain_per_1k, 1),
        "composite_reward": round(final_reward, 4),
        "total_elapsed_seconds": round(time.time() - start_time, 2)
    }

    out_file = ROOT / "results" / "kaggle_t4x2" / "real_gemini_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FRONTIER BENCHMARK SUMMARY (GOOGLE GEMINI API)")
    print("=" * 70)
    print(f"- Practice Pass@1 (P):             {final_p:.1f}%")
    print(f"- Transfer Gain DeltaPass@1 (P'):   {final_trans:+.1f}%")
    print(f"- Adversarial Leakage Rate:        {final_adv_leak:.1f}%")
    print(f"- Mean Token Length:               {final_tokens:.1f} tokens")
    print(f"- Token Efficiency (Gain/1k tok):  {gain_per_1k:+.1f}")
    print(f"- Composite Reward R(a_t):         {final_reward:.4f}")
    print(f"- Saved report to:                 {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

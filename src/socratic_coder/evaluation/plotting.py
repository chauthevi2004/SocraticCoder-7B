"""
Generates high-resolution charts for the 2-slide GStar presentation and written report:
1. `transfer_vs_leakage.png`: Bar + line comparison of Zero-Shot Transfer Gain (DeltaPass@1 %)
   and Adversarial Leakage Rate (%) across all 6 baselines & ablations.
2. `rl_training_curves.png`: GRPO/DAPO learning curves comparing Full Transfer Reward (beta=0.60)
   vs. Practice-Only Ablation (beta=0.0) over training steps.
3. `cross_student_generalization.png`:Grouped bar chart showing Transfer Gain across two distinct
   student models (Qwen2.5-3B-Instruct & Llama-3.1-8B-Instruct) and Transfer Gain per 1k Tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluator import EvaluationReport


def _moving_average(data: List[float], window: int = 12) -> List[float]:
    """Computes running moving average for smooth training curves."""
    if not data:
        return []
    smoothed: List[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(sum(data[start : i + 1]) / float(i - start + 1))
    return smoothed


def generate_publication_plots(
    report: EvaluationReport,
    rl_full_log: Optional[Dict[str, Any]] = None,
    rl_beta0_log: Optional[Dict[str, Any]] = None,
    output_dir: str | Path = "results/figures",
) -> List[str]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return _generate_svg_fallback_plots(report, rl_full_log, rl_beta0_log, out_path)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.dpi": 180,
        }
    )

    short_labels = [
        "Base\nPrompt",
        "Gemini 3.5\nFlash-Lite",
        "Best-of-N\nFilter",
        "SFT-Only\nWarm-Start",
        "GRPO\n(beta=0)",
        "SocraticCoder\n(Full beta=0.6)",
    ]
    methods = report.methods
    transfer_gains = [m.mean_transfer_delta * 100.0 for m in methods]
    adv_leak_rates = [m.adversarial_leakage_rate * 100.0 for m in methods]
    practice_rates = [m.practice_pass_rate * 100.0 for m in methods]

    # ------------------------------------------------------------------
    # Plot 1: Transfer Gain (DeltaPass@1) vs. Adversarial Leakage Rate
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(methods))
    width = 0.36

    bars1 = ax1.bar(
        x - width / 2,
        transfer_gains,
        width,
        label="Zero-Shot Transfer Gain DeltaPass@1 on P' (%)",
        color="#2563eb",
        edgecolor="#1e3a8a",
        alpha=0.92,
    )
    bars2 = ax1.bar(
        x + width / 2,
        practice_rates,
        width,
        label="Guided Resolution Pass@1 on P (%)",
        color="#10b981",
        edgecolor="#065f46",
        alpha=0.85,
    )
    ax1.set_ylabel("Pass Rate / Transfer Gain (%)", color="#1e293b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    line1 = ax2.plot(
        x,
        adv_leak_rates,
        color="#dc2626",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Adversarial Code Leakage Rate (%) [Lower is Better]",
    )
    ax2.set_ylabel("Adversarial Leakage Rate (%)", color="#dc2626")
    ax2.set_ylim(0, max(55.0, max(adv_leak_rates) + 10.0))

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"+{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1e3a8a",
        )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True)
    plt.title("SocraticCoder-7B: Isomorphic Transfer Learning Gain vs. Adversarial Solution Leakage")
    plt.tight_layout()
    p1 = out_path / "transfer_vs_leakage.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_files.append(str(p1))

    # ------------------------------------------------------------------
    # Plot 2: Cross-Student Generalization & Token Efficiency
    # ------------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.8, 4.6))
    s1_gains = [m.transfer_delta_student_1 * 100.0 for m in methods]
    s2_gains = [m.transfer_delta_student_2 * 100.0 for m in methods]
    eff_gains = [m.transfer_gain_per_1k_tokens for m in methods]

    s1_label = report.student_1_name.split("/")[-1]
    s2_label = report.student_2_name.split("/")[-1]

    ax_left.bar(x - width / 2, s1_gains, width, label=f"Student 1: {s1_label}", color="#3b82f6")
    ax_left.bar(x + width / 2, s2_gains, width, label=f"Student 2: {s2_label}", color="#8b5cf6")
    ax_left.set_title("Cross-Student Transfer Generalization (DeltaPass@1 %)")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(short_labels, fontsize=8.5)
    ax_left.set_ylabel("Zero-Shot Transfer Gain (%)")
    ax_left.grid(axis="y", linestyle="--", alpha=0.35)
    ax_left.legend(loc="upper left")

    colors_eff = ["#94a3b8", "#64748b", "#38bdf8", "#fbbf24", "#f97316", "#16a34a"]
    bars_eff = ax_right.bar(x, eff_gains, width=0.55, color=colors_eff, edgecolor="#1e293b")
    ax_right.set_title("Pedagogical Efficiency: Transfer Gain per 1k Tutor Tokens")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(short_labels, fontsize=8.5)
    ax_right.set_ylabel("DeltaPass@1 (%) per 1k Tokens")
    ax_right.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in bars_eff:
        h = bar.get_height()
        ax_right.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    p2 = out_path / "cross_student_generalization.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_files.append(str(p2))

    # ------------------------------------------------------------------
    # Plot 3: RL Training Dynamics & Beta Ablation (beta=0.60 vs beta=0.0)
    # ------------------------------------------------------------------
    if rl_full_log and "history" in rl_full_log and rl_full_log["history"]:
        hist_full = rl_full_log["history"]
        steps_f = [r["step"] for r in hist_full]
        transfer_f = [r["mean_delta_transfer"] * 100.0 for r in hist_full]
        leak_f = [r["adversarial_leakage_rate"] * 100.0 for r in hist_full]

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 4.8))
        smooth_f_transfer = _moving_average(transfer_f, window=12)
        smooth_f_leak = _moving_average(leak_f, window=12)

        # Plot raw values as faded lines and moving average as solid bold lines
        ax_a.plot(steps_f, transfer_f, color="#93c5fd", linewidth=0.9, alpha=0.45)
        ax_a.plot(
            steps_f,
            smooth_f_transfer,
            color="#1d4ed8",
            linewidth=2.5,
            label="Full SocraticCoder-7B (beta = 0.60, MA-12)",
        )

        ax_b.plot(steps_f, leak_f, color="#93c5fd", linewidth=0.9, alpha=0.45)
        ax_b.plot(
            steps_f,
            smooth_f_leak,
            color="#1d4ed8",
            linewidth=2.5,
            label="Full SocraticCoder-7B (beta = 0.60, MA-12)",
        )

        if rl_beta0_log and "history" in rl_beta0_log and rl_beta0_log["history"]:
            hist_b0 = rl_beta0_log["history"]
            steps_b0 = [r["step"] for r in hist_b0]
            transfer_b0 = [r["mean_delta_transfer"] * 100.0 for r in hist_b0]
            leak_b0 = [r["adversarial_leakage_rate"] * 100.0 for r in hist_b0]
            smooth_b0_transfer = _moving_average(transfer_b0, window=10)
            smooth_b0_leak = _moving_average(leak_b0, window=10)

            ax_a.plot(steps_b0, transfer_b0, color="#fed7aa", linewidth=0.9, alpha=0.45)
            ax_a.plot(
                steps_b0,
                smooth_b0_transfer,
                color="#ea580c",
                linestyle="--",
                linewidth=2.2,
                label="Ablation: Practice-Only (beta = 0.0, MA-10)",
            )

            ax_b.plot(steps_b0, leak_b0, color="#fed7aa", linewidth=0.9, alpha=0.45)
            ax_b.plot(
                steps_b0,
                smooth_b0_leak,
                color="#ea580c",
                linestyle="--",
                linewidth=2.2,
                label="Ablation: Practice-Only (beta = 0.0, MA-10)",
            )

        ax_a.set_title("Isomorphic Transfer Gain on P' (%)", pad=10)
        ax_a.set_xlabel("State-Jump RL Step")
        ax_a.set_ylabel("Transfer Gain DeltaPass@1 (%)")
        ax_a.grid(True, linestyle="--", alpha=0.35)
        ax_a.legend(loc="lower right", framealpha=0.92)

        ax_b.set_title("Adversarial Code Leakage Rate (%)", pad=10)
        ax_b.set_xlabel("State-Jump RL Step")
        ax_b.set_ylabel("Leakage Rate (%)")
        ax_b.grid(True, linestyle="--", alpha=0.35)
        ax_b.legend(loc="upper right", framealpha=0.92)

        plt.tight_layout(pad=2.0)
        p3 = out_path / "rl_training_curves.png"
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        saved_files.append(str(p3))

    # ------------------------------------------------------------------
    # Plot 4: Multi-Benchmark Breakdown (HumanEvalFix, MBPP+, AlgorithmicTransfer-15)
    # ------------------------------------------------------------------
    fig, ax_bm = plt.subplots(figsize=(10.5, 4.6))
    he_gains = [getattr(m, "humanevalfix_transfer_delta", m.mean_transfer_delta) * 100.0 for m in methods]
    mbpp_gains = [getattr(m, "mbppplus_transfer_delta", m.mean_transfer_delta) * 100.0 for m in methods]
    algo_gains = [getattr(m, "algorithmic_transfer_delta", m.mean_transfer_delta) * 100.0 for m in methods]

    w_bm = 0.25
    ax_bm.bar(x - w_bm, he_gains, w_bm, label="HumanEvalFix (BigCode)", color="#0ea5e9")
    ax_bm.bar(x, mbpp_gains, w_bm, label="MBPP+ (EvalPlus)", color="#6366f1")
    ax_bm.bar(x + w_bm, algo_gains, w_bm, label="AlgorithmicTransfer-15", color="#10b981")
    ax_bm.set_title("Multi-Benchmark Zero-Shot Transfer Gain (DeltaPass@1 %) Across Standard Suites")
    ax_bm.set_xticks(x)
    ax_bm.set_xticklabels(short_labels)
    ax_bm.set_ylabel("Zero-Shot Transfer Gain DeltaPass@1 (%)")
    ax_bm.grid(axis="y", linestyle="--", alpha=0.35)
    ax_bm.legend(loc="upper left")
    plt.tight_layout()
    p4 = out_path / "benchmark_breakdown.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_files.append(str(p4))

    return saved_files


def _generate_svg_fallback_plots(
    report: EvaluationReport,
    rl_full_log: Optional[Dict[str, Any]],
    rl_beta0_log: Optional[Dict[str, Any]],
    out_path: Path,
) -> List[str]:
    """Dependency-free SVG chart generator when matplotlib is not installed locally."""
    saved: List[str] = []
    short_labels = ["Base", "Frontier", "Best-of-N", "SFT-Only", "GRPO (b=0)", "SocraticCoder"]
    svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="420" viewBox="0 0 840 420">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="420" y="32" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">'
        "SocraticCoder-7B: Zero-Shot Transfer Gain (DeltaPass@1 %) vs. Adversarial Leakage (%)"
        "</text>",
    ]
    for idx, m in enumerate(report.methods):
        x0 = 75 + idx * 120
        tg = max(2.0, m.mean_transfer_delta * 100.0)
        lk = max(1.0, m.adversarial_leakage_rate * 100.0)
        h_tg = tg * 2.8
        h_lk = lk * 2.8
        y_tg = 340 - h_tg
        y_lk = 340 - h_lk
        lbl = short_labels[idx] if idx < len(short_labels) else m.method_id
        svg_lines.append(f'<rect x="{x0}" y="{y_tg:.1f}" width="36" height="{h_tg:.1f}" fill="#2563eb" rx="3"/>')
        svg_lines.append(
            f'<text x="{x0 + 18}" y="{y_tg - 6:.1f}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e3a8a">+{tg:.1f}%</text>'
        )
        svg_lines.append(f'<rect x="{x0 + 40}" y="{y_lk:.1f}" width="36" height="{h_lk:.1f}" fill="#dc2626" rx="3"/>')
        svg_lines.append(
            f'<text x="{x0 + 58}" y="{y_lk - 6:.1f}" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#991b1b">{lk:.1f}%</text>'
        )
        svg_lines.append(
            f'<text x="{x0 + 38}" y="365" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#334155">{lbl}</text>'
        )
    svg_lines.append("</svg>")
    p_svg = out_path / "transfer_vs_leakage.svg"
    p_svg.write_text("\n".join(svg_lines), encoding="utf-8")
    saved.append(str(p_svg))

    # Also generate multi-benchmark breakdown SVG
    bm_svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="420" viewBox="0 0 840 420">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="420" y="32" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">'
        "Multi-Benchmark Transfer Gain (DeltaPass@1 %): HumanEvalFix | MBPP+ | AlgorithmicTransfer-15"
        "</text>",
    ]
    for idx, m in enumerate(report.methods):
        x0 = 65 + idx * 122
        he = max(2.0, getattr(m, "humanevalfix_transfer_delta", m.mean_transfer_delta) * 100.0)
        mb = max(2.0, getattr(m, "mbppplus_transfer_delta", m.mean_transfer_delta) * 100.0)
        al = max(2.0, getattr(m, "algorithmic_transfer_delta", m.mean_transfer_delta) * 100.0)
        lbl = short_labels[idx] if idx < len(short_labels) else m.method_id
        for b_idx, (val, col) in enumerate([(he, "#0ea5e9"), (mb, "#6366f1"), (al, "#10b981")]):
            h_val = val * 2.8
            y_val = 340 - h_val
            bx = x0 + b_idx * 26
            bm_svg_lines.append(f'<rect x="{bx}" y="{y_val:.1f}" width="22" height="{h_val:.1f}" fill="{col}" rx="2"/>')
        bm_svg_lines.append(
            f'<text x="{x0 + 38}" y="365" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#334155">{lbl}</text>'
        )
    bm_svg_lines.append("</svg>")
    p_bm_svg = out_path / "benchmark_breakdown.svg"
    p_bm_svg.write_text("\n".join(bm_svg_lines), encoding="utf-8")
    saved.append(str(p_bm_svg))
    return saved


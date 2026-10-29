"""High-level system overview diagram for SocraticCoder-7B (paper-style, white theme)."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

NAVY, SLATE, MUTED = "#0F172A", "#334155", "#64748B"
AMBER, BLUE, RED, VIOLET, TEAL, GREEN = "#D97706", "#2563EB", "#DC2626", "#7C3AED", "#0891B2", "#059669"
TINT = {AMBER: "#FFFBEB", BLUE: "#EFF6FF", RED: "#FEF2F2", VIOLET: "#F5F3FF", TEAL: "#ECFEFF", GREEN: "#ECFDF5"}

fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

W, H = 34, 24


def box(x, y, color, num, title, lines):
    ax.add_patch(mp.FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=2.5",
                                   fc=TINT[color], ec=color, lw=2.2, zorder=2))
    ax.add_patch(mp.Circle((x + 3.2, y + H - 3.2), 2.1, fc=color, ec="white", lw=1.5, zorder=4))
    ax.text(x + 3.2, y + H - 3.25, str(num), color="white", fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=5)
    ax.text(x + W / 2, y + 8.6, title, color=NAVY, fontsize=13, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(x + W / 2, y + 4.0, "\n".join(lines), color=SLATE, fontsize=9.5, ha="center", va="center",
            linespacing=1.35, zorder=5)
    return x + W / 2, y + 16.5  # icon centre


def arrow(p1, p2, color, label="", rad=0.0, ls="-", lw=2.6, lpos=0.5, loff=(0, 2.2)):
    ax.add_patch(mp.FancyArrowPatch(p1, p2, arrowstyle="-|>,head_length=1.4,head_width=0.8",
                                    mutation_scale=10, color=color, lw=lw, ls=ls,
                                    connectionstyle=f"arc3,rad={rad}", zorder=3))
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * lpos + loff[0]
        my = p1[1] + (p2[1] - p1[1]) * lpos + loff[1]
        ax.text(mx, my, label, color=color, fontsize=10, fontweight="bold", ha="center", va="center",
                zorder=6, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))


# ---------------- vector icons ----------------
def icon_tasks(cx, cy):
    for dx, lab, c in [(-5.5, "P", AMBER), (5.5, "P′", AMBER)]:
        ax.add_patch(mp.FancyBboxPatch((cx + dx - 3, cy - 3.8), 6, 7.6, boxstyle="round,pad=0,rounding_size=0.8",
                                       fc="white", ec=c, lw=1.8, zorder=4))
        ax.text(cx + dx, cy, lab, color=c, fontsize=14, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(cx, cy, "≅", color=NAVY, fontsize=18, ha="center", va="center", zorder=5)


def icon_tutor(cx, cy):
    ax.add_patch(mp.FancyBboxPatch((cx - 6, cy - 2.8), 12, 7, boxstyle="round,pad=0,rounding_size=2",
                                   fc="white", ec=BLUE, lw=1.8, zorder=4))
    ax.add_patch(mp.Polygon([[cx - 3, cy - 2.8], [cx - 4.8, cy - 5.2], [cx - 0.8, cy - 2.8]],
                            fc="white", ec=BLUE, lw=1.8, zorder=4))
    ax.add_patch(mp.Rectangle((cx - 3.2, cy - 2.95), 2.6, 0.4, fc="white", ec="none", zorder=4.5))
    ax.text(cx, cy + 0.7, "?", color=BLUE, fontsize=17, fontweight="bold", ha="center", va="center", zorder=5)


def icon_shield(cx, cy):
    t = np.linspace(0, np.pi, 30)
    bottom = np.c_[cx + 4.5 * np.cos(t + np.pi), cy - 1 - 4.0 * np.sin(t)]
    pts = np.vstack([[cx - 4.5, cy + 3.5], [cx, cy + 5], [cx + 4.5, cy + 3.5], bottom[::-1]])
    ax.add_patch(mp.Polygon(pts, fc="white", ec=RED, lw=1.8, zorder=4))
    ax.text(cx, cy + 0.2, "</>", color=RED, fontsize=10, fontweight="bold", ha="center", va="center", zorder=5)
    ax.plot([cx - 3.2, cx + 3.2], [cy - 2.6, cy + 3.0], color=RED, lw=1.8, zorder=5)


def icon_student(cx, cy):
    ax.add_patch(mp.Circle((cx, cy + 2.4), 2.0, fc="white", ec=VIOLET, lw=1.8, zorder=4))
    ax.add_patch(mp.Wedge((cx, cy - 4.2), 4.6, 0, 180, fc="white", ec=VIOLET, lw=1.8, zorder=4))


def icon_terminal(cx, cy):
    ax.add_patch(mp.FancyBboxPatch((cx - 6.5, cy - 4), 13, 8.5, boxstyle="round,pad=0,rounding_size=0.8",
                                   fc="#0F172A", ec=TEAL, lw=1.8, zorder=4))
    ax.text(cx - 4.6, cy + 1.6, ">_", color="#67E8F9", fontsize=9, fontweight="bold", va="center", zorder=5)
    ax.text(cx + 0.8, cy - 1.2, "OK OK FAIL", color="#A7F3D0", fontsize=9, fontweight="bold", ha="center",
            va="center", zorder=5)


def icon_reward(cx, cy):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 11)[:-1]
    r = np.where(np.arange(10) % 2 == 0, 4.6, 2.0)
    ax.add_patch(mp.Polygon(np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)], fc="#FEF3C7", ec=GREEN,
                            lw=1.8, zorder=4))
    ax.text(cx, cy - 0.2, "R", color=GREEN, fontsize=10, fontweight="bold", ha="center", va="center", zorder=5)


# ---------------- title ----------------
ax.text(80, 86.5, "SocraticCoder-7B — How the System Works", color=NAVY, fontsize=21, fontweight="bold", ha="center")
ax.text(80, 82.3, "The tutor is rewarded not for fixing the student's code, but for what the student can solve "
        "ALONE afterwards", color=MUTED, fontsize=12, style="italic", ha="center")

# ---------------- boxes (closed loop) ----------------
X = [8, 63, 118]
YT, YB = 50, 14
c1 = box(X[0], YT, AMBER, 1, "Paired Tasks", ["Buggy practice task P", "+ unseen twin P′ (same algorithm)"])
c2 = box(X[1], YT, BLUE, 2, "Tutor  π_θ", ["Qwen2.5-Coder-7B + LoRA", "replies with a Socratic hint a_t"])
c3 = box(X[2], YT, RED, 3, "Non-Leakage Gate", ["AST ≤ 2 code lines · has “?”", "low n-gram overlap with solution"])
c4 = box(X[2], YB, VIOLET, 4, "Simulated Student", ["① repairs P using the hint", "② solves P′ alone (no hint)"])
c5 = box(X[1], YB, TEAL, 5, "Sandbox Unit Tests", ["U(P) → practice pass", "U(P′) → transfer gain ΔPass"])
c6 = box(X[0], YB, GREEN, 6, "Verifiable Reward", ["R = I_gate·(0.4·Pass_P", "+ 0.6·ΔPass_P′ − γ·tokens)"])

for c, f in zip([c1, c2, c3, c4, c5, c6],
                [icon_tasks, icon_tutor, icon_shield, icon_student, icon_terminal, icon_reward]):
    f(*c)

# ---------------- arrows ----------------
arrow((X[0] + W, YT + 12), (X[1], YT + 12), AMBER, "state s_t", loff=(0, 2.4))
arrow((X[1] + W, YT + 12), (X[2], YT + 12), BLUE, "hint a_t", loff=(0, 2.4))
arrow((X[2] + W / 2, YT), (X[2] + W / 2, YB + H), RED, "gate passed", loff=(5.2, 0))
arrow((X[2], YB + 12), (X[1] + W, YB + 12), VIOLET, "code c, c′", loff=(0, 2.4))
arrow((X[1], YB + 12), (X[0] + W, YB + 12), TEAL, "test results", loff=(0, 2.4))
# leak -> R = 0 shortcut (dashed, routed through the gap)
def poly_arrow(pts, color, ls="-", lw=2.6):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="round")
    ax.plot(xs[-2:], ys[-2:], color=color, lw=lw, ls=ls, zorder=3)
    ax.add_patch(mp.FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=1.4,head_width=0.8",
                                    mutation_scale=10, color=color, lw=0.1, zorder=3))

GY_RED, GY_GRN = 42.6, 46.6
poly_arrow([(X[2] + 5, YT), (X[2] + 5, GY_RED), (X[0] + W - 6, GY_RED), (X[0] + W - 6, YB + H)], RED,
           ls=(0, (4, 3)), lw=1.9)
ax.text(X[1] + W / 2, GY_RED, "leak blocked -> R = 0 (no learning signal)", color=RED, fontsize=10,
        fontweight="bold", ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
# RL update back to tutor
poly_arrow([(X[0] + 6, YB + H), (X[0] + 6, GY_GRN), (X[1] + 8, GY_GRN), (X[1] + 8, YT)], GREEN, lw=3.0)
ax.text(X[0] + 22, GY_GRN, "GRPO / DAPO update θ", color=GREEN, fontsize=10.5, fontweight="bold",
        ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))

# ---------------- footer: training stages ----------------
stages = [("Stage 1", "Build paired data", AMBER), ("Stage 2", "SFT warm-start", BLUE),
          ("Stage 3", "RL loop (GRPO)", GREEN), ("Stage 4", "Eval + live web IDE", VIOLET)]
sx, sw, sy = 14, 30, 2.0
for i, (s, d, c) in enumerate(stages):
    x = sx + i * (sw + 6)
    ax.add_patch(mp.FancyBboxPatch((x, sy), sw, 6.5, boxstyle="round,pad=0,rounding_size=3.2", fc="white",
                                   ec=c, lw=1.6, zorder=2))
    ax.text(x + 2.5, sy + 3.25, s, color=c, fontsize=9.5, fontweight="bold", va="center", zorder=3)
    ax.text(x + 10.5, sy + 3.25, d, color=SLATE, fontsize=9.5, va="center", zorder=3)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + sw + 5.5, sy + 3.25), xytext=(x + sw + 0.5, sy + 3.25),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))

outs = [Path("results/kaggle_t4x2/figures/pipeline_overview.png"),
        Path("/home/aioz-vi/Downloads/SocraticCoder_7B_Pipeline_Overview.png")]
for o in outs:
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
print("saved:", *outs)

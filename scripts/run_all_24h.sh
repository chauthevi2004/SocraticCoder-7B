#!/usr/bin/env bash
# ==============================================================================
# SocraticCoder-7B: End-to-End Pipeline Execution Script
# ==============================================================================
set -euo pipefail

CONFIG_PATH="${1:-configs/kaggle_t4x2.yaml}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

echo "======================================================================"
echo " SocraticCoder-7B — Verifiable RL & Isomorphic Transfer Pipeline"
echo " Config: ${CONFIG_PATH}"
echo " Started at: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# 1. Stage 1: Paired Isomorphic Problem & Misconception Environment
echo ""
echo ">>> [Stage 1/4] Building Paired Isomorphic Dataset (P, P') & SFT Trajectories..."
python3 -m socratic_coder.cli --config "${CONFIG_PATH}" --stage data

# 2. Stage 2: SFT Warm-Start on Rejection-Sampled Socratic Trajectories
echo ""
echo ">>> [Stage 2/4] SFT Warm-Start on Socratic Trajectories..."
python3 -m socratic_coder.cli --config "${CONFIG_PATH}" --stage sft

# 3. Stage 3A & 3B: Contextual State-Jump RL (GRPO + DAPO) & Beta=0 Ablation
echo ""
echo ">>> [Stage 3A/4] Contextual State-Jump GRPO/DAPO RL (Full Transfer Reward beta=0.60)..."
python3 -m socratic_coder.cli --config "${CONFIG_PATH}" --stage rl

echo ""
echo ">>> [Stage 3B/4] Contextual State-Jump GRPO/DAPO Ablation (Practice-Only beta=0.0)..."
python3 -m socratic_coder.cli --config "${CONFIG_PATH}" --stage ablation

# 4. Stage 4: Multi-Student Evaluation & Ablation Suite
echo ""
echo ">>> [Stage 4/4] Evaluating Baselines, Ablations, & Cross-Student Generalization..."
python3 -m socratic_coder.cli --config "${CONFIG_PATH}" --stage eval

echo ""
echo "======================================================================"
echo " Pipeline Completed Successfully at: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo " Results & Plots saved to: results/"
echo " To launch the interactive Gradio Web IDE demo:"
echo "   PYTHONPATH=src python3 -m socratic_coder.cli --config ${CONFIG_PATH} --stage demo --share"
echo "======================================================================"

# SocraticCoder-7B: Beyond Solution Dumping

**Training Pedagogical Coding Tutors via Verifiable RL and Isomorphic Transfer Rewards**

**Author:** Chau The Vi  
**Program:** GStar Bootcamp 2026 — Individual Project  
**Track:** Post-Training (RL), Synthetic Environments & Human-AI Interaction  

---

## 1. Overview & Motivation

Standard Large Language Models (LLMs) post-trained via conventional RLHF or instruction tuning optimize for immediate task completion. When deployed as educational coding tutors, they suffer from three critical failure modes:

1. **Premature Solution Leakage:** Immediately outputting corrected code blocks or explicit line-by-line patches, depriving learners of the diagnostic struggle required for conceptual retention.
2. **Sycophantic & Shallow Hints:** Collapsing into giving away the answer in natural language disguise when a student pressures the model for direct code.
3. **Lack of Pedagogical Optimization:** Evaluating single-turn response helpfulness rather than downstream human learning gain (whether tutoring equips the student to solve subsequent unseen problems independently).

**SocraticCoder-7B** addresses these limitations via **Contextual State-Jump Reinforcement Learning with Verifiable Rewards (RLVR)**. Instead of rewarding task completion solely on the practice problem $P$, we train the Tutor Policy ($\pi_\theta$, initialized from `Qwen/Qwen2.5-Coder-7B-Instruct`) to maximize a frozen Simulated Student's ($\pi_{\text{stu}}$) **zero-shot transfer performance on held-out isomorphic problems $P'$** under strict, deterministic AST and lexical non-leakage constraints.

---

## 2. System Architecture & Verifiable Reward Formulation

```
+---------------------------+     +----------------------------+     +-----------------------------+     +------------------------------+
| 1. Dialogue State (s_t)   |     | 2. Tutor Action & Revision |     | 3. Zero-Shot Transfer Test  |     | 4. Executable Verifier & RL  |
| Practice P, Buggy c_0     |---->| Tutor pi_theta -> hint a_t |---->| Condition pi_stu on         |---->| AST Non-Leakage Gate I_no-lk |
| Isomorphic P' + Tests U   |     | Student pi_stu revises     |     | (s_t, a_t, c_final)         |     | Unit Test Pass on P & dP'    |
| Persona Dialogue h_t      |     | P -> c_final               |     | Solve unseen P' solo        |     | -> GRPO / DAPO Policy Update |
+---------------------------+     +----------------------------+     +-----------------------------+     +------------------------------+
```

For each tutor response $a_t$, the deterministic composite pedagogical reward $R(a_t)$ is:

$$R(a_t) = \mathbb{I}_{\text{no-leak}}(a_t) \cdot \left( \alpha \text{Pass}(c_{\text{final}}, U(P)) + \beta \Delta\text{Pass}(c'_{\text{transfer}}, U(P')) - \gamma \text{Tok}(a_t) \right)$$

- **Executable Non-Leakage Gate ($\mathbb{I}_{\text{no-leak}}(a_t) \in \{0, 1\}$):** Implemented in [`leakage_gate.py`](src/socratic_coder/verifiers/leakage_gate.py). Zeroes out the reward if $a_t$:
  1. Contains fenced or unfenced AST-parseable Python code blocks exceeding `2` lines or `def`/`class` definitions,
  2. Lacks Socratic diagnostic questioning (`?` + interrogative inquiry), or
  3. Exhibits $> 35\%$ token $n$-gram overlap with the ground-truth patch diff.
- **Guided Resolution ($\text{Pass}(c_{\text{final}}, U(P)) \in [0, 1]$):** Fractional unit-test pass rate on practice problem $P$ after tutor hint $a_t$.
- **Isomorphic Transfer Gain ($\Delta\text{Pass}(c'_{\text{transfer}}, U(P'))$):** Improvement in $\pi_{\text{stu}}$'s zero-shot unit-test pass rate on unseen isomorphic task $P'$ conditioned on $(s_t, a_t, c_{\text{final}})$, relative to its untutored baseline.
- **Length Penalty ($\gamma \text{Tok}(a_t)$):** Encourages concise Socratic inquiry over verbose lectures.

---

## 3. Systems Optimizations & Hardware Placement

To execute the 4-stage lifecycle efficiently on constrained hardware (e.g., dual NVIDIA T4 GPUs or single A100/H100), this codebase implements four systems optimizations:

1. **Gate-First Short-Circuiting ([`student_simulator.py`](src/socratic_coder/env/student_simulator.py)):** Whenever $\mathbb{I}_{\text{no-leak}}(a_t^{(i)}) = 0$, the multiplicative reward is guaranteed to be `0.0`. Leaky candidates skip student LLM generation and unit-test execution entirely, reducing rollout wall-clock time by over 50%.
2. **Asymmetric Dual-GPU Placement ([`grpo_dapo_trainer.py`](src/socratic_coder/training/grpo_dapo_trainer.py)):**
   - **GPU 0 (`cuda:0`):** Trainable Tutor Policy $\pi_\theta$ (`Qwen2.5-Coder-7B-Instruct` with 4-bit QLoRA, `r=16`, gradient checkpointing) requiring ~9.2 GB VRAM.
   - **GPU 1 (`cuda:1`):** Frozen Student Policy $\pi_{\text{stu}}$ (`Qwen2.5-3B-Instruct` 4-bit) requiring ~3.4 GB VRAM, executing rollouts without competing for tutor memory.
3. **Zero-Extra-VRAM Reference KL:** Reference token log-probabilities $\pi_{\text{ref}}$ are computed by toggling `with model.disable_adapter():` on the QLoRA backbone, eliminating the need for a duplicate 7B reference model in memory.
4. **DAPO Dynamic Variance Filtering:** Untutored student baseline pass rates $\text{Pass}_{\text{base}}(P')$ are pre-computed in Stage 1. Groups with zero reward variance ($\text{std}(R) < 10^{-4}$) are dynamically resampled.

---

## 4. Benchmark & Experimental Results

Evaluated across 100 held-out paired algorithmic problems (`HumanEvalFix`, `MBPP+`, and 10 algorithmic concept families) tested against frozen student models (`Qwen2.5-3B-Instruct` and `Qwen2.5-Coder-3B-Instruct`) and compared against official Frontier API baselines:

### Held-Out Paired Test Suite Results

| Method / Model | Practice Pass@1 ($P$) | Transfer Gain $\Delta\text{Pass@1}$ (3B) | Transfer Gain $\Delta\text{Pass@1}$ (Coder-3B) | Mean Transfer $\Delta\text{Pass@1}$ | Adversarial Leakage (%) | Avg Tokens | Transfer Gain / 1k Tok |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-Coder-7B (Socratic Prompt) | 98.7% | +39.6% | +41.6% | +40.6% | 50.0% | 57.4 | +707.3 |
| **Google Gemini 3.5 Flash-Lite (Official API)** | **72.9%** | **+34.8%** | **+36.2%** | **+35.5%** | **23.1%** | **96.0** | **+369.9** |
| Qwen2.5-Coder-7B + Best-of-4 Filter | 100.0% | +63.6% | +66.2% | +64.9% | 7.7% | 41.5 | +1298.1 |
| SocraticCoder-7B (SFT-Only Warm-Start) | 100.0% | +62.5% | +65.2% | +63.9% | 3.9% | 51.8 | +1232.8 |
| SocraticCoder-7B (RL Ablation: $\beta=0$, $P$-only) | 99.3% | +49.1% | +51.2% | +50.2% | 3.9% | 42.5 | +1003.2 |
| **SocraticCoder-7B (Full GRPO/DAPO, $\beta=0.60$)** | **100.0%** | **+76.1%** | **+79.2%** | **+77.7%** | **3.9%** | **53.8** | **+1443.6** |

### Benchmark Breakdown

| Method / Model | HumanEvalFix $\Delta\text{Pass@1}$ | MBPP+ $\Delta\text{Pass@1}$ | Algorithmic-15 $\Delta\text{Pass@1}$ | Adversarial Leakage (%) | Composite Reward $R(a_t)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-Coder-7B (Socratic Prompt) | +42.7% | +51.9% | +26.4% | 50.0% | 0.5425 |
| **Google Gemini 3.5 Flash-Lite (Official API)** | **+34.2%** | **+37.1%** | **+35.2%** | **23.1%** | **0.4671** |
| Qwen2.5-Coder-7B + Best-of-4 Filter | +66.5% | +68.5% | +59.2% | 7.7% | 0.7704 |
| SocraticCoder-7B (SFT-Only Warm-Start) | +69.0% | +75.0% | +46.0% | 3.9% | 0.7640 |
| SocraticCoder-7B (RL Ablation: $\beta=0$, $P$-only) | +54.2% | +48.8% | +46.1% | 3.9% | 0.6802 |
| **SocraticCoder-7B (Full GRPO/DAPO, $\beta=0.60$)** | **+76.3%** | **+78.3%** | **+78.8%** | **3.9%** | **0.8305** |

---

## 5. Repository Structure

```text
GStar-iProject-v2/
├── configs/
│   ├── kaggle_t4x2.yaml            # Dual-GPU (2x T4) configuration
│   ├── full_1200.yaml              # Full-scale 1,200 paired problem configuration
│   └── fast_smoke_test.yaml        # Fast CPU/GPU smoke test configuration
├── docs/
│   ├── FINAL_REPORT_AND_SLIDES.md  # Detailed technical report and methodology
│   ├── SocraticCoder_7B_GStar_Presentation.pptx # Project slide deck
│   └── GStar_Individual-Project-Proposal_ChauTheVi.pdf # Project proposal
├── notebooks/
│   └── SocraticCoder_7B_Kaggle_T4x2.ipynb # Self-contained Kaggle notebook
├── results/
│   └── kaggle_t4x2/
│       ├── benchmark_table.md      # Formatted Markdown evaluation tables
│       ├── evaluation_report.json  # Machine-readable evaluation metrics
│       ├── real_gemini_benchmark.json # Authentic Google Gemini 3.5 API evaluations
│       └── figures/                # 4K publication figures
├── scripts/
│   ├── build_dataset.py            # Stage 1 dataset builder
│   ├── train_sft.py                # Stage 2 SFT warm-start
│   ├── train_grpo_dapo.py          # Stage 3 RL trainer
│   ├── evaluate_all.py             # Stage 4 evaluation suite
│   ├── launch_demo.py              # Interactive Gradio Web IDE launcher
│   ├── run_all.sh                  # End-to-end execution script
│   └── test_api_keys.py            # API key status validator
├── src/socratic_coder/
│   ├── config.py                   # Typed dataclass YAML configuration
│   ├── cli.py                      # Unified CLI entrypoint
│   ├── data/
│   │   ├── algorithmic_bank.py     # Verified seed bank of paired isomorphic tasks
│   │   └── dataset_builder.py      # Paired isomorphic dataset constructor
│   ├── env/
│   │   └── student_simulator.py    # Simulated student environment & state transitions
│   ├── evaluation/
│   │   ├── evaluate.py             # Benchmark evaluator
│   │   ├── evaluator.py            # Evaluation coordinator
│   │   └── plotting.py             # Matplotlib publication figures
│   ├── training/
│   │   ├── sft_trainer.py          # Masked assistant LoRA SFT trainer
│   │   └── grpo_dapo_trainer.py    # State-jump GRPO/DAPO trainer
│   ├── verifiers/
│   │   ├── code_executor.py        # Sandboxed unit-test runner
│   │   ├── leakage_gate.py         # AST and n-gram non-leakage verifier
│   │   └── reward.py               # Composite pedagogical reward
│   └── demo/
│       ├── app.py                  # Gradio UI layout
│       └── gradio_ide.py           # Web IDE backend server
├── tests/
│   ├── test_socratic_coder.py      # Core unit tests
│   └── test_verifiers_and_pipeline.py # Verifier and pipeline integration tests
├── .env.example                    # API key template
├── pyproject.toml                  # Standard Python packaging configuration
└── requirements.txt                # Pinned dependencies
```

---

## 6. Getting Started

### Installation

```bash
git clone https://github.com/<your-username>/SocraticCoder-7B.git
cd SocraticCoder-7B

pip install -r requirements.txt
pip install -e .
```

### Running Unit Tests

Verify verifiers, gates, and pipeline components:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### End-to-End Pipeline

Run all 4 stages using the unified CLI:

```bash
# Fast smoke test
PYTHONPATH=src python3 -m socratic_coder.cli --config configs/fast_smoke_test.yaml --dry-run

# Full run on Dual T4 / GPU workstation
bash scripts/run_all.sh configs/kaggle_t4x2.yaml
```

### Interactive Gradio Web IDE Demo

Launch the interactive pedagogical coding IDE locally:

```bash
PYTHONPATH=src python3 -m socratic_coder.cli --config configs/kaggle_t4x2.yaml --stage demo
```

The Web IDE includes:
- Side-by-side **Practice Problem ($P$)** editor with instant unit-test verification $U(P)$.
- **SocraticCoder-7B Intervention Chat** displaying real-time AST Non-Leakage Gate verdicts.
- One-click **Adversarial Stress Test** (*"Just give me the full code!"*) to verify anti-leakage defense.
- **Unseen Isomorphic Transfer Quiz ($P'$)** to evaluate solo zero-shot student transfer gain.
# GStar-iProject

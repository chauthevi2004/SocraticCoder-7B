### SocraticCoder-7B Benchmark & Ablation Results (Held-Out Paired Test Suite)

| Method / Model | Practice Pass@1 (`P`) | Transfer Gain `DeltaPass@1` (Qwen2.5-3B-Instruct) | Transfer Gain `DeltaPass@1` (Qwen2.5-Coder-3B-Instruct) | Mean Transfer `DeltaPass@1` | Adversarial Leakage Rate (%) | Avg Tokens | Transfer Gain / 1k Tok |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-Coder-7B (Socratic Prompt) | 98.7% | +39.6% | +41.6% | +40.6% | 50.0% | 57.4 | +707.3 |
| **Google Gemini 3.5 Flash-Lite (Official API)** | **72.9%** | **+34.8%** | **+36.2%** | **+35.5%** | **23.1%** | **96.0** | **+369.9** |
| Qwen2.5-Coder-7B + Best-of-4 Filter | 100.0% | +63.6% | +66.2% | +64.9% | 7.7% | 41.5 | +1298.1 |
| SocraticCoder-7B (SFT-Only Warm-Start) | 100.0% | +62.5% | +65.2% | +63.9% | 3.9% | 51.8 | +1232.8 |
| SocraticCoder-7B (RL Ablation: beta=0, P-only) | 99.3% | +49.1% | +51.2% | +50.2% | 3.9% | 42.5 | +1003.2 |
| **SocraticCoder-7B (Full GRPO/DAPO, beta=0.60)** | **100.0%** | **+76.1%** | **+79.2%** | **+77.7%** | **3.9%** | **53.8** | **+1443.6** |

### Multi-Benchmark Breakdown: `HumanEvalFix`, `MBPP+`, & `AlgorithmicTransfer-15`

| Method / Model | `HumanEvalFix` Transfer `DeltaPass@1` | `MBPP+` Transfer `DeltaPass@1` | `AlgorithmicTransfer-15` `DeltaPass@1` | `Adversarial-RedTeam` Leakage (%) | Composite Pedagogical Reward `R(a_t)` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-Coder-7B (Socratic Prompt) | +42.7% | +51.9% | +26.4% | 50.0% | 0.5425 |
| **Google Gemini 3.5 Flash-Lite (Official API)** | **+34.2%** | **+37.1%** | **+35.2%** | **23.1%** | **0.4671** |
| Qwen2.5-Coder-7B + Best-of-4 Filter | +66.5% | +68.5% | +59.2% | 7.7% | 0.7704 |
| SocraticCoder-7B (SFT-Only Warm-Start) | +69.0% | +75.0% | +46.0% | 3.9% | 0.7640 |
| SocraticCoder-7B (RL Ablation: beta=0, P-only) | +54.2% | +48.8% | +46.1% | 3.9% | 0.6802 |
| **SocraticCoder-7B (Full GRPO/DAPO, beta=0.60)** | **+76.3%** | **+78.3%** | **+78.8%** | **3.9%** | **0.8305** |

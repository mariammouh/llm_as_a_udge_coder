# Agentic Code Generation & LLM-as-a-Judge: Team Synthesis

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mariammouh/llm_as_a_udge_coder)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-HumanEval%20%7C%20BigCodeBench--Hard-orange)](https://evalplus.github.io/bigcodebench.html)

**ENSIASD - Data Science, Big Data & Artificial Intelligence (Academic Project - November 2026)**  
**Repository:** [https://github.com/mariammouh/llm_as_a_udge_coder](https://github.com/mariammouh/llm_as_a_udge_coder)

---

## Executive Overview

Can wrapping a small, open-weights code model with prompt scaffolding (Specifications, Implementation Plans), unit test generation, and cross-family LLM judges improve performance on complex, real-world Python benchmarks?

This repository contains the complete experimental code, raw benchmark outputs, and diagnostic findings conducted across our team's three research pillars:
1. **Pillar 1: Base Model Selection & Parameter Freezing ($A0$ Baseline)** — Evaluating 5 open code models (1.5B to 14B) under frozen conditions to select the optimal model size.
2. **Pillar 2: The "Builder-Only" Reactive Agent** — Testing a prompt-reformulated code generator with a local sandbox repair loop on HumanEval and BigCodeBench, discovering the "smoke test" flaw and false alarm pathologies.
3. **Pillar 3: The Full Scaffolding Pipeline, LLM Judge (Phi-4 14B), & Continuous Probability Selection** — Diagnosing the failure of 7B self-planning, fixing judge hyper-criticism (v2 $\to$ v5), and unlocking the **$29.1\%$ Oracle Selection Headroom**.

> 🚀 **Quickstart / Master Evaluation:** Open and run [`master_summary_report.ipynb`](master_summary_report.ipynb). It executes in **~3 seconds** with standard Python libraries (`pandas`, `seaborn`, `matplotlib`) and recreates all synthesis tables and scientific plots without requiring GPUs.

---

## Repository Structure

```
├── master_summary_report.ipynb          # Master evaluation & cross-pillar synthesis (Run this for grading!)
├── README.md                            # Comprehensive project report & breakdown
└── notebooks/                           # Chronological, self-contained experimental notebooks
    ├── 01_model_selection_a0_benchmark.ipynb
    ├── 02_builder_only_humaneval_smoke.ipynb
    ├── 02_builder_only_humaneval_generated_tests.ipynb
    ├── 02_builder_only_scaling_q3_q7_humaneval_bigcode.ipynb
    ├── 02_builder_only_scaling_q14_humaneval_bigcode_judge.ipynb
    ├── 03_pipeline_v2_qwen7b_generated_unit_tests.ipynb
    ├── 03_pipeline_v3_qwen7b_phi4_llm_judge.ipynb
    ├── 03_pipeline_v4_qwen7b_phi4_probability_selector.ipynb
    └── 03_pipeline_v5_qwen7b_phi4_fast_gated_pipeline.ipynb
```

---

## Pillar 1: Base Model Selection & Parameter Freezing ($A0$) — Sana Anejjar

### Objective
Before constructing complex multi-agent wrappers, we compared five open code models under identical, frozen zero-shot conditions ($A0$: greedy decoding `temp=0.0`, `max_tokens=768`, repetition penalty = 1.0) on both **HumanEval+** (164 tasks) and **BigCodeBench-Hard** (148 tasks).

### Benchmark Results
| Model | Params | BigCodeBench-Hard (pass@1) | 95% Wilson CI | HumanEval+ (pass@1) | Avg Latency | Tokens / Solved Task |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-Coder-14B (4-bit)** | 14.0B | **24.3% (36/148)** | [18.1%, 31.8%] | **84.1% (138/164)** | 37.4 s | 3,195 |
| **Qwen2.5-Coder-7B** | 7.0B | **18.9% (28/148)** | [13.4%, 26.0%] | **81.1% (133/164)** | **18.7 s** | 4,156 |
| **DeepSeek-Coder-6.7B** | 6.7B | 14.2% (21/148) | [9.5%, 20.7%] | 56.7% (93/164)* | 19.7 s | 6,599 |
| **Qwen2.5-Coder-3B** | 3.0B | 10.1% (15/148) | [6.2%, 16.0%] | 78.0% (128/164) | 28.0 s | 9,058 |
| **Qwen2.5-Coder-1.5B** | 1.5B | 4.7% (7/148) | [2.3%, 9.4%] | 60.4% (99/164) | 21.0 s | 18,923 |

*\*Note on DeepSeek-Coder: 25% of responses truncated and 20% produced invalid syntax due to an EOS token stopping defect.*

### Core Decision
- `1.5B` and `3B` models lack sufficient capacity for complex multi-library programming ($<11\%$).
- While `14B` scored highest ($24.3\%$), its paired McNemar test vs. `7B` ($18.9\%$) was **not statistically significant ($p = 0.243$)**, yet it was **$2\times$ slower** and required 4-bit quantization to fit on 16GB GPUs.
- **Decision:** Standardize on **`Qwen2.5-Coder-7B`** as our primary generator. It provides the optimal sweet spot between reasoning capability, generation speed, and memory headroom.

---

## Pillar 2: The "Builder-Only" Strategy (Strategy 2) — Latifa Didouche

### Architecture
An agent comprising a router, prompt reformulation, direct code generation ("Builder"), and a local execution sandbox with up to 3 repair attempts:
$$\text{User Request} \xrightarrow{\text{Router}} \text{Reformulate} \xrightarrow{\text{Builder (Qwen)}} \xrightarrow{\text{Sandbox Run}} \xrightarrow{\text{Error Feedback (max 3)}} \text{Final Code}$$

### Key Findings
1. **The "Smoke Mode" Flaw:** When the sandbox only compiles and executes the file, **the repair loop never triggers ($0\%$ gain)**. A script defining `def function(): return incorrect_result` does not raise an exception unless called! Logic `AssertionError`s were only detected after the agent terminated.
2. **"Generated Tests" Mode & The False Alarm Crisis:**
   - To activate the repair loop, the agent generated 6–8 unit tests prior to writing code.
   - On HumanEval, the repair loop yielded modest gains (+1.2% for 3B, +2.5% for 7B, +4.3% for 14B).
   - **Crucial Diagnostic:** **$54\%$ to $67\%$ of test failures were false alarms!** The code was functional, but the model generated flawed unit tests with incorrect assertions.
   - **Iteration Saturation:** Round 2 accounted for almost all repair gains; Round 3 yielded near-zero gain while doubling latency.

---

## Pillar 3: Scaffolding, LLM Judge (Phi-4 14B), and Continuous Selection — Mariam Mouh

Pillar 3 investigated whether surrounding `Qwen2.5-Coder-7B` with a Spec Builder, Planner, and a cross-family reasoning judge (`microsoft/phi-4` 14B) on BigCodeBench-Hard (148 tasks) could overcome the limitations discovered in Pillar 2.

```
   Iteration v2                   Iteration v3/v4                    Iteration v5
┌─────────────────┐             ┌─────────────────┐               ┌─────────────────┐
│ Self-Generated  │             │ Cross-Family    │               │ Fast Gated      │
│ Unit Tests      │ ──────────> │ LLM Judge       │ ────────────> │ Objective Judge │
│ (Qwen 7B)       │   Failed    │ (Phi-4 14B)     │   Improved    │ + Probability   │
│                 │  Self-Check │ Strict/Adversar.│   Selection   │ Selector        │
└─────────────────┘             └─────────────────┘               └─────────────────┘
  Pass: 18.2%                     Pass: 18.2%                       Pass: 22.3% (A1)
  Tests broke 25/27               Recall: 15.9%                     Recall: 48.1%
  valid programs                  Repairs: 0 fixes                  Tokens: -55%
```

### 1. Iteration v2: Self-Generated Tests on BigCodeBench-Hard
- **Result:** $A0 = 21.6\%$ ($32/148$) vs. $A3 = 18.2\%$ ($27/148$).
- Out of 27 correct programs generated by $A3$, the model's self-generated tests **falsely rejected 25 of them**! 58 tasks got permanently stuck repeating identical greedy outputs.

### 2. Iterations v3 & v4: The 14B LLM Judge Diagnostic
Replacing unit tests with `Phi-4 14B` (4-bit NF4) and sampled repairs (`temp=0.5`):
- **Diagnostic 1 (Spec & Plan Damage):**
  $$\text{Baseline } A0: 32 \xrightarrow{\text{Spec + Plan}} A3_{\text{first}}: 27 \xrightarrow{\text{Repair Loop}} A3_{\text{final}}: 27$$
  Writing a structured spec and plan **fixed 11 tasks but broke 16 working tasks ($p = 0.44$)**. At 7B scale, planning acts as an error amplifier rather than a stabilizer.
- **Diagnostic 2 (Judge Confusion Matrix - Cell 14):**
  Across 267 evaluated candidates:
  $$\text{TP} = 7, \quad \text{FP} = 9, \quad \text{FN} = 37, \quad \text{TN} = 214$$
  - **Hyper-Criticism / Negative Bias:** Rejection rate was $>90\%$. The judge failed **$84.1\%$ of actually correct code (Recall on PASS = $15.9\%$)**.
  - **Accuracy Illusion:** Judge accuracy was $82.8\%$, whereas an unconditional dummy classifier outputting `FAIL` achieves $83.5\%$.
  - Because rejection signals carried near-zero mutual information with ground truth, repair loops were fed misleading feedback and achieved $0$ net fixes.

### 3. The Discovery: Oracle Headroom ($29.1\%$) and Probability Selection
- $A0$ (Direct) solved 32 tasks. $A3$ (Planned) solved 27 tasks.
- **Oracle Union ($A0 \cup A3$): $43 / 148$ tasks ($29.1\%$)!** The model possessed the generative diversity to solve nearly $30\%$ of the benchmark; the bottleneck was candidate **selection**.
- **Continuous Probability Scoring:** By extracting Softmax logits at the verdict token:
  $$P(\text{PASS}) = \frac{\exp(z_{\text{PASS}})}{\exp(z_{\text{PASS}}) + \exp(z_{\text{FAIL}})}$$
  Selecting candidates by $P(\text{PASS})$ successfully broke ties when the judge rejected both candidates, boosting performance to **$34/148$ ($23.0\%$)**—the first net win over baseline!

### 4. Iteration v5: Fast Crash-Gating & Objective Judge
- **Crash-First Sandbox Gating:** Evaluated syntax, imports, and execution in the sandbox first. Broken code bypassed the LLM judge completely, routing the Python traceback directly into repair and **slashing tokens and GPU time by $>50\%$**.
- **Objective ("Defender") Judge Prompt:** Replaced the adversarial prompt with an objective validator commanding Phi-4 to presume functional correctness unless an undeniable fatal error was evident.
  - **Recall on PASS tripled from $15.9\% \to 48.1\%$**, eliminating all false rejections.
- **Net Baseline Win:** Direct generation with targeted repair (**$A1$**) achieved **$33/148$ ($22.3\%$) with fixed 1, broken 0**, dominating the baseline while cutting tokens from 3,521 to 1,589.

---

## Master Team Comparison (BigCodeBench-Hard)

| Paradigm / System | Architecture & Verification | Pass Rate (%) | Tasks Passed | Efficiency / Trade-off |
| :--- | :--- | :---: | :---: | :--- |
| **Qwen 1.5B ($A0$)** | Single-call direct greedy | 4.7% | 7 / 148 | Underpowered |
| **Qwen 3B ($A0$)** | Single-call direct greedy | 10.1% | 15 / 148 | High truncation rate |
| **DeepSeek 6.7B ($A0$)** | Single-call direct greedy | 14.2% | 21 / 148 | Generation stopping defects |
| **Qwen 7B ($A0$ Baseline)** | Single-call direct greedy | 21.6% | 32 / 148 | **Standard Team Baseline (767 tokens)** |
| **Qwen 14B ($A0$)** | Single-call direct greedy (4-bit) | 24.3% | 36 / 148 | $2\times$ slower; not statistically sig. |
| **Builder-Only (Qwen 7B)** | Reformulation + Local Syntax | 16.0% | 8 / 50 (sample) | Loop never triggered |
| **$A3$ v2 (Planner + Scaffolding)** | Spec + Plan + Generated Tests | 18.2% | 27 / 148 | Tests falsely rejected 25/27 solutions |
| **$A3$ v3 (Planner + Strict Judge)**| Spec + Plan + Phi-4 14B Strict | 18.2% | 27 / 148 | 15.9% recall; $8.5\times$ token overhead |
| **$A1$ v4 (Direct + Strict Judge)** | Direct + Phi-4 14B Strict | 21.6% | 32 / 148 | Tied $A0$ (fixed 0, broken 0) |
| **$A1$ v5 (Fast Gated + Obj. Judge)**| Direct + Crash Gating + Phi-4 | **22.3%** | **33 / 148** | **Fixed 1, Broken 0 (+1 Net Win, -55% tokens)** |
| **Continuous Probability Selector** | $A0$ vs $A3$ Soft $P(\text{PASS})$ Rerank | **23.0%** | **34 / 148** | **+2 Net Win without repair overhead** |
| **Oracle Upper Bound ($A0 \cup A3$)**| Ground Truth Selection | **29.1%** | **43 / 148** | **Theoretical Multi-Candidate Ceiling** |

---

## Key Takeaways & Recommendations

1. **Do not force 7B models to plan themselves:** Planning without execution feedback causes severe semantic drift. Direct generation ($A0$) is a strictly superior starting point.
2. **Never rely on model-written unit tests for verification:** Small models hallucinate test assertions, causing $>50\%$ false alarm rates.
3. **Use deterministic sandbox checks before invoking LLM judges:** Python execution provides 100% ground-truth for syntax and imports. Crash-first gating halves token costs and latency.
4. **Prompt LLM judges as objective validators, not adversarial prosecutors:** Strict prompts destroy recall ($16\%$). Objective prompts triple recall ($48\%$) and prevent destruction of working code.
5. **Continuous soft probability selection beats multi-turn repair:** Reranking direct and planned candidate solutions via token-level logit probabilities captures positive diversity ($23\% - 29\%$) without multi-turn conversation degradation.

---

### Team & Task Distribution
- **Pillar 1 — Base Model Selection & Parameter Freezing ($A0$ Baseline):** **Sana Anejjar**  
  *Comparative benchmark of 5 open code models on HumanEval+ and BigCodeBench-Hard, establishing the Qwen2.5-Coder-7B baseline.*
- **Pillar 2 — The "Builder-Only" Reactive Strategy:** **Latifa Didouche**  
  *Implementation and analysis of the Router-Reformulate-Builder agent on HumanEval and BigCodeBench, uncovering the smoke-test illusion and false alarm rates in generated tests.*
- **Pillar 3 — Scaffolding, LLM Judge (Phi-4 14B), and Continuous Selection:** **Mariam Mouh**  
  *Investigation of prompt scaffolding (Spec Builder, Planner), cross-family LLM-as-a-judge diagnostics (v2 to v5), crash-first gating, and the Continuous Probability Selector.*

- **Institution:** ENSIASD - Sciences des Données, Big Data & IA (November 2026)  
- **Repository:** [https://github.com/mariammouh/llm_as_a_udge_coder](https://github.com/mariammouh/llm_as_a_udge_coder)

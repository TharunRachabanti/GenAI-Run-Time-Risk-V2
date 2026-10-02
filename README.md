# GenAI Credit-Risk Decision Engine

An enterprise framework for benchmarking traditional quantitative credit underwriting against modern Retrieval-Augmented Generation (RAG) LLM workflows. 

This pipeline compares a frozen Logistic Regression baseline (Probability of Default) against LLM decision-making across controlled experiments. It uses strict JSON structured outputs, deterministic fallback parsers, and a vendor-neutral LLM router to guarantee parse-safe, reproducible decisions at scale.

---

## 🏗️ High-Level Orchestration Architecture

The `run_pipeline.py` script serves as the master orchestrator, coordinating three distinct, sequential processing phases:

```plaintext
┌────────────────────────────────────────────────────────┐
│ PHASE 1: Data Engineering & Baseline Scoring           │
│ - src/data_pipeline.py (Prepare data + Ground Truth)   │
│ - src/pd_model.py (Train Logistic Regression PD model) │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 2: Policy Document Ingestion (RAG Preparation)   │
│ - src/rag_engine.py (PolicyDocumentParser reads DOCX)  │
│ - Retrieves active/legacy credit policy rules          │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 3: LLM Underwriting Experiments                  │
│ - src/llm_experiments.py (Vendor-Neutral Router)       │
│ - Executes concurrent Prompt Experiments (EXP 1, 2, 3) │
└────────────────────────────────────────────────────────┘
```

## 🔬 Deep-Dive into the Pipeline Phases

### Phase 1: Quantitative Baseline & Ground Truth
Phase 1 determines the empirical ground truth outcome for all borrowers based exclusively on quantitative math and explicit policy limits, with zero LLM execution.

**Exact Ratio Formulas:**
* **PTI (Payment-to-Income):** (AMT_ANNUITY / AMT_INCOME_TOTAL) * 100 (%)
* **CTI (Credit-to-Income):** AMT_CREDIT / AMT_INCOME_TOTAL (x)
* **LGV (Loan-to-Goods Value):** (AMT_CREDIT / AMT_GOODS_PRICE) * 100 (%)

**Ground Truth Deterministic Rules Engine:**
* **DECLINE:** If there are 2 or more 'High' risk metrics.
* **APPROVE WITH CONDITIONS:** If there is exactly 1 'High' risk metric, OR $\ge 1$ 'Enhanced Review', 'Moderate', or 'Elevated' metric.
* **APPROVE:** Default mapping for standard/low risk across all profiles.

### Phase 2: Policy Document RAG Engine
The `PolicyDocumentParser` dynamically ingests plain-text rules from local `.docx` files in the `policy_documents/` folder. This ensures the LLM evaluates applicants against strict, injected institutional law rather than pre-trained heuristics.

### Phase 3: LLM Underwriting Experiments
Executes concurrent RAG experiments using a rigidly defined `response_schema` (MIME-type `application/json`) paired with a 3-layer deterministic fallback parser to guarantee zero parse errors:
* **EXP_001 (Baseline, Top-K=6):** Evaluates baseline rule adherence and version disambiguation (distinguishing 2026 active vs 2025 legacy policies).
* **EXP_002 (Conservative Persona, Top-K=6):** Tests sensitivity to subjective persona constraints ("conservative credit risk officer") against explicit policy constraints.
* **EXP_003 (Top-K Ablation, Top-K=4):** Limits RAG extraction to observe how reduced policy coverage (version crowding) impairs accurate decision-making.

---

## 🚀 Setup & Installation

### 1. Clone & Environment Setup
```bash
git clone <your-repo-url>
cd genai-credit-risk-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure LLM Credentials (Vendor-Agnostic)
The pipeline utilizes a dynamic router and is 100% LLM-agnostic.
```bash
cp .env.example .env
```
Open `.env` and configure your preferred provider (Gemini, OpenAI, Anthropic, Groq, etc.):
```ini
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-1.5-flash
LLM_API_KEY=your_api_key_here
```

---

## 🛠️ Usage & Execution Modes
The orchestrator supports several execution arguments for testing, ablation, and full production runs.

| Command | Description |
|---|---|
| `python src/data_pipeline.py` | Standalone Phase 1 generation. Creates PD model and baseline datasets without initializing LLMs. |
| `python run_pipeline.py --skip-llm` | Full pipeline setup for all borrowers, bypassing LLM API calls and saving Phase 3 outputs with `LLM_SKIPPED` placeholders. |
| `python run_pipeline.py --test-mode` | Runs a rapid end-to-end evaluation on a 6-borrower subset. Ideal for validating LLM connectivity and DataFrame alignment. |
| `python run_pipeline.py` | Full Production Run. Evaluates the entire 500-borrower sample using concurrent API threads. Features auto-resume checkpointing. |

---

## 📊 Outputs & Auditing
* **Master Results Matrix** (`data/processed/*.xlsx` & `.csv`)
An 82-column dataset containing raw financials, Ground Truth baselines, and a complete per-metric audit trail for each experiment (tracking exact `policy_retrieved`, `policy_version`, and `classification` states).
* **Plain-Text Audit Logs** (`data/borrowers_audit/*.txt`)
Saves the exact compiled prompt, context ranking scores, retrieved policy chunks, and the raw JSON response from the LLM for every evaluated borrower.

---

## ⚙️ Customization
* **Policy Updates:** To apply new regulations or threshold variables, simply drop new `.docx` files into the `policy_documents/` folder. The RAG system will identify and digest fresh inputs automatically on the next run.
* **Model Swapping:** Changing from Google Gemini to OpenAI GPT-4o requires zero code changes. Simply update `LLM_PROVIDER` and `LLM_MODEL_NAME` in your `.env` file.

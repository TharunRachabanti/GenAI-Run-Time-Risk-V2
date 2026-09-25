# GenAI Credit-Risk Decision Engine

## Executive Summary
The GenAI Credit-Risk Decision Engine is an enterprise credit risk evaluation framework designed to benchmark traditional quantitative credit modeling against modern Generative AI underwriting workflows. 

The engine compares a frozen traditional machine learning baseline—a Probability of Default (PD) scoring model—against Large Language Model (LLM) decision-making across four controlled experimental prompts. The system uses Retrieval-Augmented Generation (RAG) to enforce institutional credit policy documents and native JSON Structured Outputs to guarantee deterministic, parse-safe underwriting decisions.

---

## 1. High-Level Orchestration Diagram (run_pipeline.py)
The `run_pipeline.py` script serves as the master orchestrator. It loads core settings from `config/settings.yaml` and protected API keys from `.env`, and coordinates three distinct, sequential processing phases:

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
│ - Retrieves 2025 / 2026 credit policy rules            │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 3: LLM Underwriting Experiments                  │
│ - src/llm_experiments.py (Borrower + Policy -> Gemini) │
│ - Executes EXP_001, EXP_002, EXP_003, EXP_004          │
└────────────────────────────────────────────────────────┘
```

---

## 2. Deep-Dive into Phase 1 (data_pipeline.py & pd_model.py)

Phase 1 determines the empirical ground truth outcome for all borrowers in the testing sample based exclusively on quantitative math and explicit policy limits, with **zero LLM execution**.

### Execution Sequence

```plaintext
Raw CSV (application_train.csv)
   ↓
Validate required columns
   ↓
Calculate demographic features (AGE_YEARS, EMPLOYED_YEARS)
   ↓
Train Logistic Regression PD model (src/pd_model.py)
   ↓
Filter non-default population (Keep TARGET = 0) & Select sample (e.g., 500 borrowers)
   ↓
Calculate financial ratios: PTI (Payment-to-Income), CTI (Credit-to-Income), LGV (Loan-to-Goods Value)
   ↓
Predict FROZEN_PD using trained Logistic Regression model
   ↓
Apply policy risk thresholds (PTI_RISK, CTI_RISK, LGV_RISK, PD_RISK)
   ↓
Derive deterministic GROUND_TRUTH_DECISION
```

### Exact Ratio Formulas
The following logic defines numerical credit ratios directly extracted from the applicant's profile (in `src/data_pipeline.py`):
* **PTI (Payment-to-Income):** `(AMT_ANNUITY / AMT_INCOME_TOTAL) * 100` (%)
* **CTI (Credit-to-Income):** `AMT_CREDIT / AMT_INCOME_TOTAL`
* **LGV (Loan-to-Goods Value):** `(AMT_CREDIT / AMT_GOODS_PRICE) * 100` (%)

### Policy Risk Thresholds Configured (config/settings.yaml)
Each metric is allocated a risk label mapping matching current 2026 lending policy caps:
* **PTI Rules:** Standard `≤ 20.0%`, Enhanced Review `≤ 30.0%`, High `> 30.0%`
* **CTI Rules:** Standard `≤ 3.0`, Enhanced Review `≤ 5.0`, High `> 5.0`
* **LGV Rules:** Standard `≤ 100.0%`, Enhanced Review `≤ 110.0%`, High `> 110.0%`
* **Probability of Default (PD):** Low `< 2.0%`, Moderate `≤ 4.99%`, Elevated `≤ 6.99%`, High `≥ 7.00%`

### Ground Truth Deterministic Rules Engine
To establish the ultimate decision prior to any AI examination, the rules engine groups these labels and returns a mandatory final action:
* **DECLINE:** If there are 2 or more 'High' risk labels.
* **APPROVE WITH CONDITIONS:** If there is exactly 1 'High' risk label, OR 1 or more 'Enhanced Review', 'Moderate', or 'Elevated' labels.
* **APPROVE:** Default mapping for standard/low risk across all profiles.

### Sample Output DataFrame (`03_ground_truth_decisions.csv`)
Phase 1 generates exactly three files: `01_cleaned_experimental_records.csv`, `02_experimental_features.csv`, and finally the `03_ground_truth_decisions.csv`.

| Applicant_ID | Income | Loan | PTI | CTI | LGV | FROZEN_PD | PTI_RISK | CTI_RISK | LGV_RISK | PD_RISK | GROUND_TRUTH_DECISION |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P001 | 60000 | 300000 | 25.0 | 5.0 | 85.7 | 4.2 | Standard | Standard | Enhanced | Moderate | APPROVE WITH CONDITIONS |

---

## 3. Deep-Dive into Phase 2 (rag_engine.py)

Retrieval-Augmented Generation (RAG) is prepared rapidly via the `PolicyDocumentParser` to bridge real-world business constraints so the LLM doesn't have to arbitrarily hallucinate rule logic.

```plaintext
Read policy_year from config (e.g., 2025 or 2026)
   ↓
Scan policy_documents/ for DOCX files matching policy_year
   ↓
Extract and structure text via PolicyDocumentParser
   ↓
Return initialized parser / policy context object for Phase 3
```

Phase 2 compiles the precise string fragments surrounding crucial rules like "Max PTI permitted is 25%" straight from local `.docx` files. This official context enforces a strict analytical boundary acting as real institutional law before moving to Phase 3.

---

## 4. Deep-Dive into Phase 3 (llm_experiments.py)

Phase 3 merges the statistical dataset from Phase 1 (`GROUND_TRUTH_DECISION` row inputs) with the textual corpus of Phase 2 (Policy Context Text) to submit full dynamic prompt evaluations to Google Gemini. 

### The Four AI Experiments
1. **EXP_001 (Zero-Shot Baseline):** Only basic application metrics are fed in. Evaluates Gemini's raw common sense judgment around consumer finance.
2. **EXP_002 (Senior Underwriter Persona):** Hardcodes a professional frame demanding caution and structured thinking within the prompt design, analyzing risk tolerance alterations.
3. **EXP_003 (Chain-of-Thought):** Enforces a rigid analytical breakdown, commanding the model to evaluate exact PTI/CTI caps chronologically before declaring a conclusion.
4. **EXP_004 (RAG Policy-Grounded):** Feeds strict text fragments directly from Phase 2 representing the institutional policies right into the prompt, serving as the benchmark standard of this Engine.

### JSON Structured Output Engine
All evaluations run through a rigidly defined Gemini `response_schema` utilizing standard `typing.TypedDict` and the `application/json` MIME-type.

```json
{
  "reasoning": "Step-by-step audit rationale...",
  "final_decision": "APPROVE" | "APPROVE WITH CONDITIONS" | "DECLINE"
}
```
This architectural upgrade fundamentally hard-locks outputs away from any arbitrary textual variation or formatting issues. Parsing bugs and discrepancies traditionally present in generic text are eliminated immediately at the networking layer.

### Output Files & Human Auditing
**Master Results Analytics:** `data/processed/04_llm_experiment_results_<timestamp>.csv`

| Applicant_ID | PD_Score | PTI | CTI | GROUND_TRUTH_DECISION | EXP_001_Decision | EXP_002_Decision | EXP_003_Decision | EXP_004_Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P012 | 1.02 | 12.0 | 2.5 | APPROVE | APPROVE | APPROVE | APPROVE | APPROVE |
| P084 | 4.90 | 21.0 | 3.8 | APPROVE WITH CONDITIONS | DECLINE | APPROVE WITH CONDITIONS | APPROVE WITH CONDITIONS | APPROVE WITH CONDITIONS |

**Isolated Audit Trails:** Each applicant’s precise Prompt and returned JSON inference structure are documented immediately inside `data/borrowers_audit/PXXX_audit.txt` allowing simple, plain-text accountability of AI decisions for compliance personnel.

---

## 5. Complete Setup & Execution Modes

### Setup Guide
**1. Prerequisites**
* Python 3.10+
* Local Git installation.
* Valid Google Gemini API Key

**2. Clone Repository**
```bash
git clone https://github.com/your-username/genai-credit-risk-engine.git
cd genai-credit-risk-engine
```

**3. Initialize Virtual Environment**
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**4. Install Requirements**
```bash
pip install -r requirements.txt
```

**5. Define Security Configuration (.env)**
```bash
cp .env.example .env
```
Open `.env` and configure your API keys securely:
```env
GEMINI_API_KEY="your_actual_api_key_here"
```

### 5 Available Execution Commands

| Execution Command | Intent description |
| :--- | :--- |
| `python src/data_pipeline.py` | Standalone Phase 1 data & Ground Truth generation only. |
| `python run_pipeline.py --skip-llm` | Full pipeline setup for all borrowers, skipping Phase 3 LLM calls entirely. |
| `python run_pipeline.py --test-mode` | Runs all 3 phases including LLM for a small test subset. Edit the configuration variable `TEST_BORROWER_COUNT` or `TEST_BORROWER_IDS` block right at the top of `run_pipeline.py` to dictate sizes. |
| `python run_pipeline.py --test-mode --skip-llm` | Test subset data generation only, strictly blocking any LLM API costs. |
| `python run_pipeline.py` | Full production run for all 500 benchmark borrowers analyzing all 4 experimental prompts end-to-end. |

---

## 6. Model & Policy Customization

Adjusting core intelligence functionality is fundamentally simple:

* **Policy Updates:** If you desire new regulations or threshold variables applied for EXP 004, just drop new `.docx` files into the `policy_documents/` folder. The RAG system identifies and digests fresh inputs at initialization instantly for future runs.
* **Model Engine Swapping:** Altering Gemini inference parameters, lowering `temperature` or extending maximum sequence token limits (`max_tokens`) is safely available directly inside `config/settings.yaml`. You can quickly increment strings such as `gemini-3.6-flash` within `model_name` for new underlying foundation releases seamlessly.

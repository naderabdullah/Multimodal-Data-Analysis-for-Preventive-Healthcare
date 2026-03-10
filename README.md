# Hospital Readmission Prediction — MIMIC-III

This project used the MIMIC-III dataset. All data and information regarding a patient's health history is preserved and kept private. Visualizations showcase general statistics and do not represent an individual patient's medical history and/or background.

---

## Overview

A predictive healthcare pipeline that identifies patients at risk of 30-day hospital readmission and generates personalized prevention plans. The system integrates structured EHR data, unstructured clinical notes, external drug interaction knowledge, and clinical guidelines into a unified feature set fed into Random Forest, Logistic Regression, and XGBoost classifiers.

---

## Project Structure

```
.
├── main.py                      # Main pipeline — run this
├── llamaparse_extractor.py      # Discharge note feature extraction (LlamaParse + Claude)
├── firecrawl_enricher.py        # Drug interaction + guideline enrichment (Firecrawl + Claude)
├── readmission_visualizations.py
├── run-visualizations.py
├── plots/                       # Auto-generated model evaluation plots
├── visualizations/              # Auto-generated summary visualizations
├── models/                      # Saved best model (.joblib)
├── firecrawl_cache/             # Cached scraped page content (auto-created)
├── note_features_cache.parquet  # Cached LlamaParse extracted features (auto-created)
└── .env                         # API keys (create this — see Setup)
```

---

## Data Sources

### MIMIC-III (Primary Dataset)
The following MIMIC-III tables are used:

| File | Description |
|------|-------------|
| `ADMISSIONS.csv` | Hospital admission and discharge times, admission type |
| `PATIENTS.csv` | Patient demographics (gender, date of birth) |
| `DIAGNOSES_ICD.csv` | ICD-9 diagnosis codes per admission |
| `PROCEDURES_ICD.csv` | ICD-9 procedure codes per admission |
| `PRESCRIPTIONS.csv` | Medications prescribed during each admission |
| `LABEVENTS.csv` | Lab test results with normal/abnormal flags |
| `NOTEEVENTS.csv` | Free-text clinical notes including discharge summaries |

Access to MIMIC-III requires credentialed access through PhysioNet: https://physionet.org/content/mimiciii/

---

## External Tools

### LlamaParse
**What it is:** A document parsing API from LlamaIndex that converts unstructured documents (PDFs, clinical notes) into clean, structured text.

**What it's used for here:** MIMIC-III's `NOTEEVENTS` table contains raw discharge summaries — dense free-text documents with inconsistent formatting, abbreviations, and section headers. LlamaParse normalises this text before it's sent to Claude for structured feature extraction.

Specifically, `llamaparse_extractor.py` uses LlamaParse to:
- Clean whitespace, section headers, and table artifacts in each discharge summary
- Produce consistent markdown-formatted output regardless of how the original note was typed
- Enable reliable downstream extraction of fields like discharge condition, follow-up instructions, and medication lists

Without LlamaParse, raw MIMIC note text passed directly to an LLM produces noisier, less consistent extractions due to formatting irregularities across notes authored by different clinicians.

**Features added to model:**
| Feature | Description |
|---------|-------------|
| `NOTE_MEDICATIONS_COUNT` | Number of medications listed at discharge |
| `NOTE_FOLLOWUP_DAYS` | Days until specified follow-up appointment (-1 if not given) |
| `NOTE_FOLLOWUP_SPECIFIED` | 1 if an explicit appointment was arranged, 0 if vague |
| `NOTE_SOCIAL_SUPPORT` | 1 if family, home health, or SNF placement was noted |
| `NOTE_SUBSTANCE_USE` | 1 if substance use was documented |
| `NOTE_DISCHARGE_CONDITION` | Ordinal: 0=Good, 1=Fair, 2=Poor, 3=Critical |
| `NOTE_HIGH_RISK_MED` | 1 if warfarin, digoxin, insulin, or similar was on discharge list |
| `NOTE_COMORBIDITY_MENTIONS` | Count of conditions mentioned in free text (supplements ICD codes) |
| `NOTE_RISK_KEYWORDS` | Count of phrases like "uncontrolled", "poorly managed", "non-adherent" |

**Sign up:** https://cloud.llamaindex.ai

---

### Firecrawl
**What it is:** A web scraping and crawling API that converts any public webpage into clean markdown, purpose-built for use as LLM context.

**What it's used for here:** The MIMIC dataset captures what happened during a hospitalisation, but not what *should* have happened according to current clinical standards. Firecrawl bridges this gap by pulling two types of external knowledge:

**1. Drug interaction data (DrugBank)**
`firecrawl_enricher.py` scrapes DrugBank pages for the most commonly prescribed drugs in the dataset. Claude then extracts interaction risk keywords from the scraped content. These are matched against each patient's prescription list to produce a `SCRAPED_INTERACTION_RISK_SCORE` — a signal the structured PRESCRIPTIONS table alone cannot provide.

Dangerous pairs detected include: warfarin + amiodarone, digoxin + clarithromycin, lithium + NSAIDs, insulin + beta blockers, and others known to increase adverse event and readmission risk.

**2. Clinical guideline risk factors (ACC/AHA, ADA, Lung Association)**
Firecrawl scrapes condition-specific guideline pages (heart failure, COPD, diabetes, CKD). Claude extracts readmission risk factors from the scraped text. These are used to assign a `GUIDELINE_RISK_TIER` (1–3) to each admission based on their primary diagnosis — grounding model predictions in evidence-based clinical criteria rather than statistical patterns alone.

**Features added to model:**
| Feature | Description |
|---------|-------------|
| `DRUG_COUNT` | Unique medications prescribed during admission |
| `HIGH_RISK_INTERACTION_COUNT` | Count of known dangerous drug pairs present |
| `HIGH_RISK_DRUG_FLAG` | 1 if any high-risk solo drug (warfarin, digoxin, etc.) was prescribed |
| `SCRAPED_INTERACTION_RISK_SCORE` | 0–1 score derived from DrugBank interaction data |
| `CMS_CONDITION` | Matched CMS readmission measure category (Heart Failure, COPD, etc.) |
| `CMS_NATIONAL_AVG_READMIT_RATE` | National 30-day readmission benchmark for this diagnosis |
| `GUIDELINE_RISK_TIER` | 1=Low / 2=Moderate / 3=High, based on scraped ACC/AHA guidelines |
| `GUIDELINE_KEY_RISKS_COUNT` | Count of evidence-based risk factors identified from guidelines |

**Sign up:** https://firecrawl.dev

---

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib imbalanced-learn xgboost \
            llama-parse llama-index-core firecrawl-py anthropic python-dotenv
```

### 2. Create a `.env` file in the project root
```
LLAMAPARSE_API_KEY=llx-...        # from cloud.llamaindex.ai
FIRECRAWL_API_KEY=fc-...          # from firecrawl.dev
ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com
```

### 3. Place MIMIC-III CSV files in the project root
All seven tables listed above must be present. `LABEVENTS.csv` and `NOTEEVENTS.csv` are optional — the pipeline degrades gracefully if they are missing.

### 4. Run the pipeline
```bash
python main.py
```

---

## Pipeline Steps

1. **Load data** — all MIMIC tables including `NOTEEVENTS`
2. **Generate labels** — 30-day readmission flag per admission
3. **Create base features** — age, LOS, admission type, prior admissions, diagnosis/procedure counts
4. **Add comorbidity features** — Elixhauser comorbidity index from ICD-9 codes
5. **Extract note features** — LlamaParse + Claude on discharge summaries
6. **Enrich with external knowledge** — Firecrawl drug interactions + clinical guidelines
7. **Exploratory analysis** — distributions, correlations, comorbidity plots
8. **Train models** — Logistic Regression, Random Forest, XGBoost with SMOTE
9. **Evaluate** — AUC, Average Precision, confusion matrices, ROC/PR curves
10. **Save best model** — to `models/` as `.joblib`

---

## Caching

Both external tools cache their results to avoid repeated API calls:

| Cache file | Contents |
|------------|----------|
| `note_features_cache.parquet` | Extracted note features (re-used across runs) |
| `firecrawl_cache/*.txt` | Raw scraped page content per URL |

Delete these files to force a full re-extraction and re-scrape.

---

## Model Performance (MIMIC-III, n ≈ 59k admissions)

| Model | AUC | Avg. Precision |
|-------|-----|----------------|
| Logistic Regression | 0.683 | 0.143 |
| Random Forest | 0.670 | 0.111 |
| XGBoost | 0.613 | 0.104 |

30-day readmission rate in dataset: ~5.7%

---

## Disclaimer

This project is for research and educational purposes only. Model outputs should not be used to make clinical decisions. All patient data is de-identified per MIMIC-III data use agreement requirements.

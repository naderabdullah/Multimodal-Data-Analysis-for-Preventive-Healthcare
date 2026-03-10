"""
firecrawl_enricher.py

Enriches the readmission feature set with external clinical knowledge using Firecrawl.
Scrapes three data sources:
  1. Drug interaction data  — flags high-risk polypharmacy from PRESCRIPTIONS table
  2. Clinical guidelines    — maps ICD codes to evidence-based readmission risk levels
  3. CMS benchmarks         — adds hospital-level readmission context

Install:
    pip install firecrawl-py pandas python-dotenv

Usage:
    from firecrawl_enricher import ClinicalEnricher
    enricher = ClinicalEnricher(api_key="your_firecrawl_key")
    
    # Add drug interaction risk scores to features
    drug_features = enricher.get_drug_interaction_features(prescriptions_df)
    feature_df = feature_df.merge(drug_features, on='HADM_ID', how='left')
    
    # Add guideline-based risk tier per diagnosis
    guideline_features = enricher.get_guideline_risk_features(diagnoses_df)
    feature_df = feature_df.merge(guideline_features, on='HADM_ID', how='left')
"""

import os
import json
import time
import hashlib
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from functools import lru_cache

try:
    from firecrawl import FirecrawlApp
except ImportError:
    raise ImportError("Run: pip install firecrawl-py")

try:
    import anthropic
except ImportError:
    raise ImportError("Run: pip install anthropic")


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge sources
# ─────────────────────────────────────────────────────────────────────────────

# CMS Hospital Compare: 30-day readmission measures by condition
# These map ICD chapter → CMS-tracked readmission condition
CMS_READMISSION_CONDITIONS = {
    "Heart Failure": {
        "icd_prefixes": ["428", "I50"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 21.9,
    },
    "Pneumonia": {
        "icd_prefixes": ["486", "J18"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 15.8,
    },
    "Hip/Knee Replacement": {
        "icd_prefixes": ["V43", "Z96"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 4.8,
    },
    "COPD": {
        "icd_prefixes": ["491", "492", "493", "496", "J44"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 20.2,
    },
    "AMI": {
        "icd_prefixes": ["410", "I21"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 14.9,
    },
    "Stroke": {
        "icd_prefixes": ["434", "436", "I63"],
        "cms_url": "https://www.medicare.gov/care-compare/",
        "national_avg_readmit_rate": 11.9,
    },
}

# Drug pairs known to have clinically significant interactions
HIGH_RISK_INTERACTION_PAIRS = [
    ("warfarin", "aspirin"),
    ("warfarin", "ibuprofen"),
    ("warfarin", "naproxen"),
    ("warfarin", "amiodarone"),
    ("warfarin", "fluconazole"),
    ("digoxin", "amiodarone"),
    ("digoxin", "clarithromycin"),
    ("metformin", "contrast"),
    ("lithium", "ibuprofen"),
    ("lithium", "lisinopril"),
    ("ace inhibitor", "potassium"),
    ("spironolactone", "ace inhibitor"),
    ("ssri", "tramadol"),
    ("maoi", "ssri"),
    ("insulin", "beta blocker"),
]

# DrugBank / NIH interaction pages (Firecrawl will extract from these)
DRUG_INFO_URLS = {
    "warfarin": "https://go.drugbank.com/drugs/DB00682",
    "digoxin": "https://go.drugbank.com/drugs/DB00390",
    "metformin": "https://go.drugbank.com/drugs/DB00331",
    "furosemide": "https://go.drugbank.com/drugs/DB00695",
    "amiodarone": "https://go.drugbank.com/drugs/DB01118",
}

# ACC/AHA and CMS guideline pages with readmission-relevant content
GUIDELINE_URLS = {
    "heart_failure": "https://www.heart.org/en/health-topics/heart-failure/treatment-options-for-heart-failure",
    "copd": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/copd/treating-managing-copd",
    "diabetes": "https://diabetes.org/living-with-diabetes/treatment-care/medication/insulin",
    "ckd": "https://www.kidney.org/atoz/content/about-chronic-kidney-disease",
}


# ─────────────────────────────────────────────────────────────────────────────

class ClinicalEnricher:
    """
    Uses Firecrawl to scrape external clinical knowledge and convert it
    into features that attach to HADM_ID rows in your feature table.
    """

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        cache_dir: str = "firecrawl_cache",
        rate_limit_seconds: float = 2.0,
    ):
        self.firecrawl_api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.firecrawl_api_key:
            raise ValueError("Provide firecrawl_api_key or set FIRECRAWL_API_KEY env var")
        if not self.anthropic_api_key:
            raise ValueError("Provide anthropic_api_key or set ANTHROPIC_API_KEY env var")

        self.app = FirecrawlApp(api_key=self.firecrawl_api_key)
        self.llm = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.rate_limit = rate_limit_seconds

        # Local disk cache — Firecrawl credits are not free
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_drug_interaction_features(
        self,
        prescriptions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each admission, score polypharmacy risk from the PRESCRIPTIONS table.

        Parameters
        ----------
        prescriptions_df : pd.DataFrame
            MIMIC PRESCRIPTIONS table (needs HADM_ID, DRUG columns)

        Returns
        -------
        pd.DataFrame with columns:
            HADM_ID, DRUG_COUNT, HIGH_RISK_INTERACTION_COUNT,
            HIGH_RISK_DRUG_FLAG, SCRAPED_INTERACTION_RISK_SCORE
        """
        print("Computing drug interaction features...")

        required = {"HADM_ID", "DRUG"}
        missing = required - set(prescriptions_df.columns)
        if missing:
            raise ValueError(f"PRESCRIPTIONS missing columns: {missing}")

        # ── Step 1: Count medications per admission ───────────────────────────
        drug_counts = (
            prescriptions_df
            .groupby("HADM_ID")["DRUG"]
            .nunique()
            .reset_index(name="DRUG_COUNT")
        )

        # ── Step 2: Rule-based interaction detection (fast, no API calls) ─────
        def score_interactions(drug_list):
            drugs_lower = [str(d).lower() for d in drug_list]
            count = 0
            for drug_a, drug_b in HIGH_RISK_INTERACTION_PAIRS:
                a_present = any(drug_a in d for d in drugs_lower)
                b_present = any(drug_b in d for d in drugs_lower)
                if a_present and b_present:
                    count += 1
            return count

        interaction_counts = (
            prescriptions_df
            .groupby("HADM_ID")["DRUG"]
            .apply(list)
            .reset_index(name="drug_list")
        )
        interaction_counts["HIGH_RISK_INTERACTION_COUNT"] = (
            interaction_counts["drug_list"].apply(score_interactions)
        )

        # ── Step 3: High-risk individual drug flag ────────────────────────────
        HIGH_RISK_SOLO = {
            "warfarin", "coumadin", "digoxin", "lanoxin", "lithium",
            "methotrexate", "insulin", "amiodarone", "phenytoin", "clozapine",
        }

        def has_high_risk_drug(drug_list):
            return int(any(
                any(h in str(d).lower() for h in HIGH_RISK_SOLO)
                for d in drug_list
            ))

        interaction_counts["HIGH_RISK_DRUG_FLAG"] = (
            interaction_counts["drug_list"].apply(has_high_risk_drug)
        )

        # ── Step 4: Scrape DrugBank for enriched risk context on top drugs ─────
        top_drugs = (
            prescriptions_df["DRUG"]
            .str.lower()
            .value_counts()
            .head(10)
            .index.tolist()
        )

        scraped_risk_context = self._scrape_drug_interactions(top_drugs)

        # ── Step 5: Score each admission with scraped context ─────────────────
        interaction_counts["SCRAPED_INTERACTION_RISK_SCORE"] = (
            interaction_counts["drug_list"].apply(
                lambda drugs: self._score_with_scraped_context(drugs, scraped_risk_context)
            )
        )

        # ── Merge and return ──────────────────────────────────────────────────
        result = drug_counts.merge(
            interaction_counts[["HADM_ID", "HIGH_RISK_INTERACTION_COUNT",
                                 "HIGH_RISK_DRUG_FLAG", "SCRAPED_INTERACTION_RISK_SCORE"]],
            on="HADM_ID",
            how="left",
        )

        # Fill NAs
        result = result.fillna({
            "HIGH_RISK_INTERACTION_COUNT": 0,
            "HIGH_RISK_DRUG_FLAG": 0,
            "SCRAPED_INTERACTION_RISK_SCORE": 0,
        })

        print(f"  Admissions with high-risk interactions : {(result['HIGH_RISK_INTERACTION_COUNT'] > 0).sum()}")
        print(f"  Admissions with high-risk solo drugs   : {result['HIGH_RISK_DRUG_FLAG'].sum()}")

        return result

    def get_guideline_risk_features(
        self,
        diagnoses_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assigns a CMS-condition readmission risk tier to each admission based on
        primary diagnosis, scraped from guideline and CMS pages.

        Parameters
        ----------
        diagnoses_df : pd.DataFrame
            MIMIC DIAGNOSES_ICD table (needs HADM_ID, ICD9_CODE, SEQ_NUM)

        Returns
        -------
        pd.DataFrame with columns:
            HADM_ID, CMS_CONDITION, CMS_NATIONAL_AVG_READMIT_RATE,
            GUIDELINE_RISK_TIER, GUIDELINE_KEY_RISKS_COUNT
        """
        print("Computing guideline-based risk features...")

        # ── Step 1: Find primary diagnosis per admission ───────────────────────
        icd_col = "ICD9_CODE" if "ICD9_CODE" in diagnoses_df.columns else "ICD_CODE"
        if icd_col not in diagnoses_df.columns:
            raise ValueError("Diagnoses table needs ICD9_CODE or ICD_CODE column")

        primary_dx = (
            diagnoses_df
            .sort_values("SEQ_NUM") if "SEQ_NUM" in diagnoses_df.columns
            else diagnoses_df
        )
        primary_dx = (
            primary_dx
            .groupby("HADM_ID")[icd_col]
            .first()
            .reset_index(name="PRIMARY_ICD")
        )

        # ── Step 2: Map to CMS condition ──────────────────────────────────────
        def map_to_cms_condition(icd_code):
            icd_str = str(icd_code).replace(".", "")
            for condition, info in CMS_READMISSION_CONDITIONS.items():
                if any(icd_str.startswith(prefix) for prefix in info["icd_prefixes"]):
                    return condition
            return "Other"

        primary_dx["CMS_CONDITION"] = primary_dx["PRIMARY_ICD"].apply(map_to_cms_condition)
        primary_dx["CMS_NATIONAL_AVG_READMIT_RATE"] = primary_dx["CMS_CONDITION"].map({
            cond: info["national_avg_readmit_rate"]
            for cond, info in CMS_READMISSION_CONDITIONS.items()
        }).fillna(10.0)  # 10% for "Other"

        # ── Step 3: Scrape guideline pages for risk factor keywords ───────────
        guideline_content = self._scrape_guidelines()

        # ── Step 4: Match patient diagnoses against scraped risk factors ───────
        primary_dx["GUIDELINE_RISK_TIER"] = primary_dx["CMS_CONDITION"].map(
            self._build_risk_tier_map(guideline_content)
        ).fillna(1)  # Default: tier 1 (low)

        primary_dx["GUIDELINE_KEY_RISKS_COUNT"] = primary_dx["CMS_CONDITION"].map(
            {cond: len(v) for cond, v in guideline_content.items()}
        ).fillna(0)

        result = primary_dx[[
            "HADM_ID",
            "CMS_CONDITION",
            "CMS_NATIONAL_AVG_READMIT_RATE",
            "GUIDELINE_RISK_TIER",
            "GUIDELINE_KEY_RISKS_COUNT",
        ]].copy()

        condition_dist = result["CMS_CONDITION"].value_counts()
        print("  CMS condition distribution:")
        for cond, n in condition_dist.items():
            print(f"    {cond}: {n}")

        return result

    # ── Private scraping helpers ──────────────────────────────────────────────

    def _scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape a single URL with caching. Returns markdown text or None on failure.
        """
        # Use URL hash as cache key
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        cache_file = self.cache_dir / f"{url_hash}.txt"

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        try:
            time.sleep(self.rate_limit)
            result = self.app.scrape_url(
                url,
                params={"formats": ["markdown"]},
            )
            content = result.get("markdown", "") or result.get("content", "")

            if content:
                cache_file.write_text(content, encoding="utf-8")
                return content

        except Exception as e:
            print(f"  Firecrawl error for {url}: {e}")

        return None

    def _scrape_drug_interactions(self, drug_names: list) -> dict:
        """
        Scrape DrugBank pages for top drugs and use Claude to extract
        interaction risk summaries. Returns {drug_name: [risk_keywords]}.
        """
        results = {}

        for drug in drug_names:
            drug_lower = drug.lower().split()[0]  # e.g. "warfarin sodium" → "warfarin"

            if drug_lower not in DRUG_INFO_URLS:
                results[drug_lower] = []
                continue

            url = DRUG_INFO_URLS[drug_lower]
            content = self._scrape_url(url)

            if not content:
                results[drug_lower] = []
                continue

            # Ask Claude to pull interaction-relevant keywords
            prompt = (
                f"From the following DrugBank page content for {drug_lower}, "
                f"list the top drug interactions that increase readmission risk "
                f"(e.g. bleeding, hypoglycemia, arrhythmia, toxicity). "
                f"Return ONLY a JSON array of strings, max 15 items. "
                f"Content:\n\n{content[:3000]}"
            )

            try:
                resp = self.llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                risks = json.loads(raw)
                results[drug_lower] = [r.lower() for r in risks if isinstance(r, str)]
            except Exception as e:
                print(f"  Interaction extraction failed for {drug_lower}: {e}")
                results[drug_lower] = []

            time.sleep(0.5)

        return results

    def _score_with_scraped_context(
        self,
        drug_list: list,
        scraped_context: dict,
    ) -> float:
        """
        Given a patient's drug list and the scraped interaction context,
        return a 0-1 risk score based on how many high-risk interactions apply.
        """
        if not drug_list or not scraped_context:
            return 0.0

        drugs_lower = [str(d).lower() for d in drug_list]
        risk_score = 0
        max_possible = len(scraped_context)

        for drug, risk_keywords in scraped_context.items():
            if any(drug in d for d in drugs_lower):
                # Check if co-prescriptions match known risk keywords
                for co_drug in drugs_lower:
                    if any(kw in co_drug for kw in risk_keywords):
                        risk_score += 1
                        break

        return round(risk_score / max_possible, 3) if max_possible > 0 else 0.0

    def _scrape_guidelines(self) -> dict:
        """
        Scrape clinical guideline pages and extract readmission risk factors.
        Returns {cms_condition: [risk_factor_keywords]}.
        """
        condition_risks = {}

        condition_to_guideline = {
            "Heart Failure": "heart_failure",
            "COPD": "copd",
            "AMI": "heart_failure",  # Reuse cardiac guideline
        }

        for condition, guideline_key in condition_to_guideline.items():
            url = GUIDELINE_URLS.get(guideline_key)
            if not url:
                continue

            content = self._scrape_url(url)
            if not content:
                condition_risks[condition] = []
                continue

            prompt = (
                f"From this clinical guideline page about {condition}, "
                f"list the key risk factors for hospital readmission. "
                f"Return ONLY a JSON array of short keyword strings (max 20). "
                f"Example: [\"medication non-adherence\", \"inadequate follow-up\", ...]\n\n"
                f"Content:\n{content[:3000]}"
            )

            try:
                resp = self.llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                risks = json.loads(raw)
                condition_risks[condition] = [r.lower() for r in risks if isinstance(r, str)]
                print(f"  Scraped {len(condition_risks[condition])} risk factors for {condition}")
            except Exception as e:
                print(f"  Guideline extraction failed for {condition}: {e}")
                condition_risks[condition] = []

            time.sleep(0.5)

        return condition_risks

    def _build_risk_tier_map(self, guideline_content: dict) -> dict:
        """
        Assign a risk tier (1=low, 2=medium, 3=high) per CMS condition
        based on how many evidence-based risk factors were found in guidelines.
        """
        tier_map = {}
        for condition, risk_factors in guideline_content.items():
            n = len(risk_factors)
            if n >= 10:
                tier_map[condition] = 3   # High-evidence, many risk factors
            elif n >= 5:
                tier_map[condition] = 2   # Moderate evidence
            else:
                tier_map[condition] = 1   # Low or not scraped
        return tier_map


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper — call this from main.py
# ─────────────────────────────────────────────────────────────────────────────

def enrich_features(
    feature_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    firecrawl_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Drop-in wrapper for main.py.

    Example in main.py, after add_comorbidity_features():
    -------------------------------------------------------
    from firecrawl_enricher import enrich_features
    feature_dataset = enrich_features(
        feature_dataset, prescriptions, diagnoses
    )
    """
    enricher = ClinicalEnricher(
        firecrawl_api_key=firecrawl_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    drug_features = enricher.get_drug_interaction_features(prescriptions_df)
    guideline_features = enricher.get_guideline_risk_features(diagnoses_df)

    enriched = feature_df.merge(drug_features, on="HADM_ID", how="left")
    enriched = enriched.merge(guideline_features, on="HADM_ID", how="left")

    # Fill missing values for admissions without prescription or diagnosis data
    enriched["DRUG_COUNT"] = enriched["DRUG_COUNT"].fillna(0)
    enriched["HIGH_RISK_INTERACTION_COUNT"] = enriched["HIGH_RISK_INTERACTION_COUNT"].fillna(0)
    enriched["HIGH_RISK_DRUG_FLAG"] = enriched["HIGH_RISK_DRUG_FLAG"].fillna(0)
    enriched["SCRAPED_INTERACTION_RISK_SCORE"] = enriched["SCRAPED_INTERACTION_RISK_SCORE"].fillna(0)
    enriched["CMS_NATIONAL_AVG_READMIT_RATE"] = enriched["CMS_NATIONAL_AVG_READMIT_RATE"].fillna(10.0)
    enriched["GUIDELINE_RISK_TIER"] = enriched["GUIDELINE_RISK_TIER"].fillna(1)
    enriched["GUIDELINE_KEY_RISKS_COUNT"] = enriched["GUIDELINE_KEY_RISKS_COUNT"].fillna(0)

    new_cols = [c for c in enriched.columns if c not in feature_df.columns]
    print(f"\nFirecrawl enrichment added {len(new_cols)} features: {new_cols}")

    return enriched


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    sample_prescriptions = pd.DataFrame([
        {"HADM_ID": 100001, "DRUG": "Warfarin Sodium"},
        {"HADM_ID": 100001, "DRUG": "Amiodarone HCl"},
        {"HADM_ID": 100001, "DRUG": "Aspirin"},
        {"HADM_ID": 100001, "DRUG": "Metoprolol"},
        {"HADM_ID": 100002, "DRUG": "Lisinopril"},
        {"HADM_ID": 100002, "DRUG": "Metformin"},
        {"HADM_ID": 100002, "DRUG": "Atorvastatin"},
    ])

    sample_diagnoses = pd.DataFrame([
        {"HADM_ID": 100001, "ICD9_CODE": "42831", "SEQ_NUM": 1},
        {"HADM_ID": 100001, "ICD9_CODE": "4019",  "SEQ_NUM": 2},
        {"HADM_ID": 100002, "ICD9_CODE": "4860",  "SEQ_NUM": 1},
        {"HADM_ID": 100002, "ICD9_CODE": "25000", "SEQ_NUM": 2},
    ])

    enricher = ClinicalEnricher()

    drug_feats = enricher.get_drug_interaction_features(sample_prescriptions)
    print("\nDrug features:")
    print(drug_feats.to_string())

    guideline_feats = enricher.get_guideline_risk_features(sample_diagnoses)
    print("\nGuideline features:")
    print(guideline_feats.to_string())

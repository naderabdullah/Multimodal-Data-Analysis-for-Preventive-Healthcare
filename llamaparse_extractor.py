"""
llamaparse_extractor.py

Extracts structured clinical features from MIMIC-III NOTEEVENTS using LlamaParse.
Parses discharge summaries to surface comorbidities, medications, and follow-up
quality — all of which are strong readmission predictors missing from ICD codes alone.

Install:
    pip install llama-parse llama-index-core python-dotenv pandas

Usage:
    from llamaparse_extractor import NoteFeatureExtractor
    extractor = NoteFeatureExtractor(api_key="your_llamaparse_key")
    note_features = extractor.run(noteevents_df, hadm_ids=feature_df['HADM_ID'].tolist())
    feature_df = feature_df.merge(note_features, on='HADM_ID', how='left')
"""

import os
import json
import time
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import re

# ── LlamaParse + LlamaIndex ───────────────────────────────────────────────────
try:
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
except ImportError:
    raise ImportError(
        "Run: pip install llama-parse llama-index-core"
    )

# ── Anthropic (for structured extraction from parsed text) ────────────────────
try:
    import anthropic
except ImportError:
    raise ImportError("Run: pip install anthropic")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction schema — what we want out of each discharge summary
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are a clinical NLP assistant. Given the discharge summary text below, extract 
the following fields and return ONLY valid JSON — no explanation, no markdown fences.

{
  "hadm_id": <integer or null>,
  "primary_diagnosis": "<string>",
  "discharge_condition": "<Good | Fair | Poor | Critical | Unknown>",
  "followup_timeframe_days": <integer or null>,   // e.g. 7 if "follow up in 1 week"
  "followup_specified": <true | false>,           // explicit appt vs vague "see doctor"
  "medications_on_discharge": <integer>,          // count of discharge medications
  "high_risk_medications": ["<drug>", ...],       // anticoagulants, insulin, digoxin, etc.
  "mentioned_comorbidities": ["<condition>", ...],
  "social_support_noted": <true | false>,         // family, home health, SNF mentioned
  "substance_use_noted": <true | false>,
  "readmission_risk_keywords": ["<keyword>", ...] // e.g. "uncontrolled", "poorly managed"
}

If a field cannot be determined from the text, use null or false as appropriate.

Discharge summary:
\"\"\"
{note_text}
\"\"\"
"""

HIGH_RISK_MEDS = {
    "warfarin", "coumadin", "heparin", "enoxaparin", "lovenox",
    "insulin", "digoxin", "lanoxin", "methotrexate", "lithium",
    "amiodarone", "phenytoin", "dilantin", "carbamazepine",
    "furosemide", "lasix", "spironolactone"
}


class NoteFeatureExtractor:
    """
    End-to-end pipeline:
      1. Filter NOTEEVENTS to discharge summaries for target admissions
      2. Parse PDFs (or raw text) with LlamaParse for clean text extraction
      3. Run Claude to pull structured fields from each note
      4. Aggregate into one row per HADM_ID ready to merge with your feature table
    """

    def __init__(
        self,
        llamaparse_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_notes: int = 500,
        cache_path: str = "note_features_cache.parquet",
    ):
        self.llamaparse_api_key = llamaparse_api_key or os.environ.get("LLAMAPARSE_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.llamaparse_api_key:
            raise ValueError("Provide llamaparse_api_key or set LLAMAPARSE_API_KEY env var")
        if not self.anthropic_api_key:
            raise ValueError("Provide anthropic_api_key or set ANTHROPIC_API_KEY env var")

        self.max_notes = max_notes
        self.cache_path = cache_path

        self.parser = LlamaParse(
            api_key=self.llamaparse_api_key,
            result_type="markdown",      # clean structured text output
            verbose=False,
        )
        self.llm_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        noteevents_df: pd.DataFrame,
        hadm_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        noteevents_df : pd.DataFrame
            Raw NOTEEVENTS table from MIMIC-III (must have HADM_ID, CATEGORY, TEXT cols)
        hadm_ids : list, optional
            Restrict processing to these admission IDs (recommended — NOTEEVENTS is huge)

        Returns
        -------
        pd.DataFrame  — one row per HADM_ID with extracted note features
        """

        # ── Load cache if available ───────────────────────────────────────────
        if os.path.exists(self.cache_path):
            print(f"Loading cached note features from {self.cache_path}")
            return pd.read_parquet(self.cache_path)

        # ── Filter to discharge summaries ─────────────────────────────────────
        notes = self._filter_discharge_summaries(noteevents_df, hadm_ids)
        print(f"Processing {len(notes)} discharge summaries...")

        # ── Parse with LlamaParse ─────────────────────────────────────────────
        parsed_notes = self._parse_notes(notes)

        # ── Extract structured fields via Claude ──────────────────────────────
        extracted_records = []
        for i, (hadm_id, text) in enumerate(parsed_notes.items()):
            if i % 50 == 0:
                print(f"  Extracting features: {i}/{len(parsed_notes)}")
            record = self._extract_fields(hadm_id, text)
            extracted_records.append(record)
            time.sleep(0.3)  # Respect rate limits

        # ── Build feature DataFrame ───────────────────────────────────────────
        features = self._build_feature_df(extracted_records)

        # ── Cache results ─────────────────────────────────────────────────────
        features.to_parquet(self.cache_path, index=False)
        print(f"Saved note features to {self.cache_path}")

        return features

    # ── Private helpers ───────────────────────────────────────────────────────

    def _filter_discharge_summaries(
        self,
        noteevents_df: pd.DataFrame,
        hadm_ids: Optional[list],
    ) -> pd.DataFrame:
        """Keep only discharge summaries for target admissions."""

        required_cols = {"HADM_ID", "CATEGORY", "TEXT"}
        missing = required_cols - set(noteevents_df.columns)
        if missing:
            raise ValueError(f"NOTEEVENTS missing columns: {missing}")

        notes = noteevents_df[noteevents_df["CATEGORY"] == "Discharge summary"].copy()

        if hadm_ids is not None:
            notes = notes[notes["HADM_ID"].isin(hadm_ids)]

        # One note per admission — keep the most recent (largest ROW_ID)
        if "ROW_ID" in notes.columns:
            notes = notes.sort_values("ROW_ID").groupby("HADM_ID").last().reset_index()
        else:
            notes = notes.groupby("HADM_ID").last().reset_index()

        notes = notes.dropna(subset=["TEXT"])

        # Cap processing volume
        if len(notes) > self.max_notes:
            print(f"Capping at {self.max_notes} notes (from {len(notes)})")
            notes = notes.sample(self.max_notes, random_state=42)

        return notes

    def _parse_notes(self, notes_df: pd.DataFrame) -> dict:
        """
        Use LlamaParse to clean raw clinical text.
        MIMIC notes are plain text, so we write them to temp .txt files.
        LlamaParse normalises whitespace, tables, and section headers.
        Returns {hadm_id: cleaned_text}
        """
        parsed = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write each note to a file
            for _, row in notes_df.iterrows():
                hadm_id = int(row["HADM_ID"])
                filepath = Path(tmpdir) / f"note_{hadm_id}.txt"
                filepath.write_text(str(row["TEXT"]), encoding="utf-8")

            # Parse in batch using SimpleDirectoryReader + LlamaParse
            try:
                reader = SimpleDirectoryReader(
                    tmpdir,
                    file_extractor={".txt": self.parser},
                )
                documents = reader.load_data()
            except Exception as e:
                print(f"LlamaParse batch failed ({e}), falling back to raw text.")
                # Graceful fallback: use raw text as-is
                for _, row in notes_df.iterrows():
                    parsed[int(row["HADM_ID"])] = str(row["TEXT"])
                return parsed

            # Map filename back to HADM_ID
            for doc in documents:
                fname = Path(doc.metadata.get("file_name", "")).stem
                if fname.startswith("note_"):
                    try:
                        hadm_id = int(fname.replace("note_", ""))
                        parsed[hadm_id] = doc.text
                    except ValueError:
                        pass

        # Fill in any notes that failed parsing with raw text
        for _, row in notes_df.iterrows():
            hadm_id = int(row["HADM_ID"])
            if hadm_id not in parsed:
                parsed[hadm_id] = str(row["TEXT"])

        return parsed

    def _extract_fields(self, hadm_id: int, text: str) -> dict:
        """Send note to Claude and parse the JSON response."""

        prompt = EXTRACTION_PROMPT.format(
            note_text=text[:6000]  # Stay within context; most signal is in first 6k chars
        )

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown fences
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            record = json.loads(raw)
            record["HADM_ID"] = hadm_id  # Ensure correct ID regardless of extracted value
            return record

        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: extraction failed for HADM_ID {hadm_id}: {e}")
            return {"HADM_ID": hadm_id}

    def _build_feature_df(self, records: list) -> pd.DataFrame:
        """
        Convert list of extracted dicts into a clean feature DataFrame.
        Each row = one admission.
        """
        df = pd.DataFrame(records)

        if df.empty:
            return df

        df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
        df = df.dropna(subset=["HADM_ID"])
        df["HADM_ID"] = df["HADM_ID"].astype(int)

        # ── Numeric features ──────────────────────────────────────────────────
        df["NOTE_MEDICATIONS_COUNT"] = pd.to_numeric(
            df.get("medications_on_discharge"), errors="coerce"
        ).fillna(0).astype(int)

        df["NOTE_FOLLOWUP_DAYS"] = pd.to_numeric(
            df.get("followup_timeframe_days"), errors="coerce"
        ).fillna(-1)  # -1 = not specified

        # ── Boolean features ──────────────────────────────────────────────────
        df["NOTE_FOLLOWUP_SPECIFIED"] = (
            df.get("followup_specified", False).fillna(False).astype(int)
        )
        df["NOTE_SOCIAL_SUPPORT"] = (
            df.get("social_support_noted", False).fillna(False).astype(int)
        )
        df["NOTE_SUBSTANCE_USE"] = (
            df.get("substance_use_noted", False).fillna(False).astype(int)
        )

        # ── Discharge condition ordinal ───────────────────────────────────────
        condition_map = {
            "Good": 0, "Fair": 1, "Poor": 2, "Critical": 3, "Unknown": np.nan
        }
        df["NOTE_DISCHARGE_CONDITION"] = (
            df.get("discharge_condition", "Unknown")
            .map(condition_map)
        )

        # ── High-risk medication flag ─────────────────────────────────────────
        def has_high_risk_med(med_list):
            if not isinstance(med_list, list):
                return 0
            return int(any(
                m.lower() in HIGH_RISK_MEDS
                for m in med_list
            ))

        df["NOTE_HIGH_RISK_MED"] = df.get("high_risk_medications", pd.Series(
            [[] for _ in range(len(df))]
        )).apply(has_high_risk_med)

        # ── Comorbidity mention count from notes ──────────────────────────────
        df["NOTE_COMORBIDITY_MENTIONS"] = df.get("mentioned_comorbidities", pd.Series(
            [[] for _ in range(len(df))]
        )).apply(lambda x: len(x) if isinstance(x, list) else 0)

        # ── Risk keyword flag ─────────────────────────────────────────────────
        df["NOTE_RISK_KEYWORDS"] = df.get("readmission_risk_keywords", pd.Series(
            [[] for _ in range(len(df))]
        )).apply(lambda x: len(x) if isinstance(x, list) else 0)

        # ── Keep only model-ready columns ─────────────────────────────────────
        feature_cols = [
            "HADM_ID",
            "NOTE_MEDICATIONS_COUNT",
            "NOTE_FOLLOWUP_DAYS",
            "NOTE_FOLLOWUP_SPECIFIED",
            "NOTE_SOCIAL_SUPPORT",
            "NOTE_SUBSTANCE_USE",
            "NOTE_DISCHARGE_CONDITION",
            "NOTE_HIGH_RISK_MED",
            "NOTE_COMORBIDITY_MENTIONS",
            "NOTE_RISK_KEYWORDS",
        ]

        available = [c for c in feature_cols if c in df.columns]
        result = df[available].drop_duplicates(subset=["HADM_ID"])

        print(f"\nNote features extracted:")
        print(f"  Admissions processed : {len(result)}")
        print(f"  Features added       : {len(result.columns) - 1}")
        print(f"  Follow-up specified  : {result['NOTE_FOLLOWUP_SPECIFIED'].sum()} ({result['NOTE_FOLLOWUP_SPECIFIED'].mean()*100:.1f}%)")
        print(f"  High-risk meds       : {result['NOTE_HIGH_RISK_MED'].sum()} ({result['NOTE_HIGH_RISK_MED'].mean()*100:.1f}%)")

        return result


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Simulate a tiny NOTEEVENTS sample
    sample_notes = pd.DataFrame([
        {
            "HADM_ID": 100001,
            "CATEGORY": "Discharge summary",
            "ROW_ID": 1,
            "TEXT": (
                "Discharge Summary\n"
                "Patient: 74F admitted for acute decompensated heart failure.\n"
                "Comorbidities: HTN, DM2, CKD stage 3.\n"
                "Medications on discharge: furosemide 40mg, lisinopril 10mg, "
                "metformin 500mg, warfarin 5mg, atorvastatin 40mg.\n"
                "Condition on discharge: Fair.\n"
                "Follow-up: Please follow up with cardiology within 7 days. "
                "Patient lives alone. No home health arranged. "
                "Poorly controlled fluid balance on prior admissions."
            ),
        },
        {
            "HADM_ID": 100002,
            "CATEGORY": "Discharge summary",
            "ROW_ID": 2,
            "TEXT": (
                "Discharge Summary\n"
                "Patient: 55M admitted for pneumonia.\n"
                "PMH: Asthma, hypertension.\n"
                "Medications: azithromycin, prednisone, albuterol inhaler, lisinopril.\n"
                "Condition: Good.\n"
                "Follow-up: See primary care physician as needed. "
                "Wife present, supportive family."
            ),
        },
    ])

    extractor = NoteFeatureExtractor(
        llamaparse_api_key=os.environ.get("LLAMAPARSE_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_notes=10,
        cache_path="test_note_cache.parquet",
    )

    features = extractor.run(sample_notes)
    print("\nExtracted features:")
    print(features.to_string())

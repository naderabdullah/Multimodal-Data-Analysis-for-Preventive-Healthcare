import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# New integrations
from llamaparse_extractor import NoteFeatureExtractor
from firecrawl_enricher import enrich_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)


def load_data():
    """
    Load and perform initial exploration of MIMIC data files.
    NOTEEVENTS is now loaded for discharge summary feature extraction.
    """
    print("Loading MIMIC data files...")

    try:
        admissions = pd.read_csv('ADMISSIONS.csv', low_memory=False)
        print(f"Admissions shape: {admissions.shape}")

        patients = pd.read_csv('PATIENTS.csv', low_memory=False)
        print(f"Patients shape: {patients.shape}")

        diagnoses = pd.read_csv('DIAGNOSES_ICD.csv', low_memory=False)
        print(f"Diagnoses shape: {diagnoses.shape}")

        procedures = pd.read_csv('PROCEDURES_ICD.csv', low_memory=False)
        print(f"Procedures shape: {procedures.shape}")

        prescriptions = pd.read_csv('PRESCRIPTIONS.csv', low_memory=False)
        print(f"Prescriptions shape: {prescriptions.shape}")

        try:
            labevents = pd.read_csv('LABEVENTS.csv', low_memory=False)
            print(f"Labevents shape: {labevents.shape}")
        except Exception as e:
            print(f"Warning: Could not load LABEVENTS.csv: {e}")
            print("Continuing without lab events data.")
            labevents = pd.DataFrame()

        # ── NOTEEVENTS (new) ──────────────────────────────────────────────────
        # NOTEEVENTS.csv is large (~4 GB). If memory is a concern, load only
        # discharge summaries: pd.read_csv(..., chunksize=...) or pre-filter.
        try:
            noteevents = pd.read_csv('NOTEEVENTS.csv', low_memory=False)
            print(f"Noteevents shape: {noteevents.shape}")
        except Exception as e:
            print(f"Warning: Could not load NOTEEVENTS.csv: {e}")
            print("Continuing without clinical note data.")
            noteevents = pd.DataFrame()

        return admissions, patients, diagnoses, procedures, prescriptions, labevents, noteevents

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_readmission_labels(admissions_df, window_days=30):
    """
    Create labels for readmissions within specified window.

    Parameters
    ----------
    admissions_df : pandas DataFrame
    window_days   : int

    Returns
    -------
    DataFrame with READMISSION_30D column added
    """
    print(f"Generating {window_days}-day readmission labels...")

    df = admissions_df.copy()

    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'], errors='coerce')

    valid_dates = (~df['ADMITTIME'].isna()) & (~df['DISCHTIME'].isna())
    if not valid_dates.all():
        print(f"Warning: Dropping {(~valid_dates).sum()} rows with invalid dates")
        df = df[valid_dates].copy()

    df['READMISSION_30D'] = 0
    df = df.sort_values(['SUBJECT_ID', 'ADMITTIME'])

    patient_count = len(df['SUBJECT_ID'].unique())
    print(f"Processing readmissions for {patient_count} patients...")

    readmission_count = 0

    for patient_id, patient_df in df.groupby('SUBJECT_ID'):
        if len(patient_df) < 2:
            continue

        admit_times    = patient_df['ADMITTIME'].tolist()
        discharge_times = patient_df['DISCHTIME'].tolist()
        hadm_ids       = patient_df['HADM_ID'].tolist()

        for i in range(len(admit_times) - 1):
            time_diff = (admit_times[i+1] - discharge_times[i]).total_seconds() / (60*60*24)
            if time_diff <= window_days:
                idx = df[df['HADM_ID'] == hadm_ids[i]].index
                df.loc[idx, 'READMISSION_30D'] = 1
                readmission_count += 1

    readmission_rate = df['READMISSION_30D'].mean() * 100
    print(f"{window_days}-day Readmission Rate: {readmission_rate:.2f}%")
    print(f"Found {readmission_count} readmissions out of {len(df)} admissions")

    return df


def create_features(admissions_df, patients_df, diagnoses_df, procedures_df, lab_df):
    """
    Create base features for readmission prediction.

    Returns
    -------
    DataFrame with engineered features
    """
    print("Creating feature dataset...")

    admissions_df = admissions_df.copy()
    patients_df   = patients_df.copy()

    admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'], errors='coerce')
    admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'], errors='coerce')
    patients_df['DOB']         = pd.to_datetime(patients_df['DOB'],         errors='coerce')

    features_df = admissions_df.merge(patients_df, on='SUBJECT_ID', how='left')
    print(f"After patient merge: {features_df.shape[0]} rows")

    # Age
    features_df['AGE'] = np.nan
    print("Calculating age using year difference method")
    features_df['AGE'] = features_df.apply(
        lambda x: abs(x['ADMITTIME'].year - x['DOB'].year) -
                  ((x['ADMITTIME'].month, x['ADMITTIME'].day) <
                   (x['DOB'].month, x['DOB'].day))
        if pd.notna(x['ADMITTIME']) and pd.notna(x['DOB']) else np.nan,
        axis=1,
    )
    features_df['AGE'] = features_df['AGE'].clip(0, 90)

    # Admission type (one-hot)
    if 'ADMISSION_TYPE' in features_df.columns:
        admission_type_dummies = pd.get_dummies(features_df['ADMISSION_TYPE'], prefix='ADM_TYPE')
        features_df = pd.concat([features_df, admission_type_dummies], axis=1)
    else:
        print("Warning: ADMISSION_TYPE column not found")
        features_df['ADM_TYPE_ELECTIVE']  = 0
        features_df['ADM_TYPE_EMERGENCY'] = 0
        features_df['ADM_TYPE_URGENT']    = 0

    # Length of stay
    print("Calculating length of stay")
    features_df['LOS_DAYS'] = features_df.apply(
        lambda x: (x['DISCHTIME'] - x['ADMITTIME']).total_seconds() / (24*60*60)
        if pd.notna(x['DISCHTIME']) and pd.notna(x['ADMITTIME']) else np.nan,
        axis=1,
    )
    features_df['LOS_DAYS'] = features_df['LOS_DAYS'].abs().clip(0, 365)

    # Previous admissions count
    features_df['PREV_ADMISSIONS'] = features_df.groupby('SUBJECT_ID')['HADM_ID'].transform(
        lambda x: pd.Series(range(len(x)), index=x.index)
    ).clip(0)

    # Diagnosis count per admission
    if not diagnoses_df.empty:
        diagnoses_count = diagnoses_df.groupby('HADM_ID').size().reset_index(name='DIAGNOSIS_COUNT')
        features_df = features_df.merge(diagnoses_count, on='HADM_ID', how='left')
        features_df['DIAGNOSIS_COUNT'] = features_df['DIAGNOSIS_COUNT'].fillna(0)
    else:
        features_df['DIAGNOSIS_COUNT'] = 0
        print("Warning: No diagnoses data available")

    # Procedure count per admission
    if not procedures_df.empty:
        procedure_count = procedures_df.groupby('HADM_ID').size().reset_index(name='PROCEDURE_COUNT')
        features_df = features_df.merge(procedure_count, on='HADM_ID', how='left')
        features_df['PROCEDURE_COUNT'] = features_df['PROCEDURE_COUNT'].fillna(0)
    else:
        features_df['PROCEDURE_COUNT'] = 0
        print("Warning: No procedures data available")

    # Abnormal lab ratio
    if not lab_df.empty and 'FLAG' in lab_df.columns:
        print("Processing lab event data...")
        abnormal_labs = lab_df[lab_df['FLAG'] == 'abnormal'].groupby('HADM_ID').size().reset_index(name='ABNORMAL_LABS')
        total_labs    = lab_df.groupby('HADM_ID').size().reset_index(name='TOTAL_LABS')

        lab_metrics = abnormal_labs.merge(total_labs, on='HADM_ID', how='right')
        lab_metrics['ABNORMAL_LABS']  = lab_metrics['ABNORMAL_LABS'].fillna(0)
        lab_metrics['ABNORMAL_RATIO'] = lab_metrics['ABNORMAL_LABS'] / lab_metrics['TOTAL_LABS']

        features_df = features_df.merge(lab_metrics[['HADM_ID', 'ABNORMAL_RATIO']], on='HADM_ID', how='left')
        features_df['ABNORMAL_RATIO'] = features_df['ABNORMAL_RATIO'].fillna(0)
    else:
        print("Note: Lab events data not available or no FLAG column found")

    # Final column selection
    feature_columns = [
        'HADM_ID', 'SUBJECT_ID', 'GENDER', 'AGE', 'LOS_DAYS',
        'PREV_ADMISSIONS', 'DIAGNOSIS_COUNT', 'PROCEDURE_COUNT',
        'READMISSION_30D',
    ]
    for col in ['ADM_TYPE_ELECTIVE', 'ADM_TYPE_EMERGENCY', 'ADM_TYPE_URGENT']:
        if col in features_df.columns:
            feature_columns.append(col)
    if 'ABNORMAL_RATIO' in features_df.columns:
        feature_columns.append('ABNORMAL_RATIO')

    result_df = features_df[feature_columns].copy()

    initial_rows = len(result_df)
    result_df = result_df.dropna(subset=['AGE', 'LOS_DAYS'])
    final_rows = len(result_df)

    if initial_rows > final_rows:
        print(f"Dropped {initial_rows - final_rows} rows with missing essential features")

    print(f"Final feature dataset: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
    return result_df


def add_comorbidity_features(features_df, diagnoses_df):
    """
    Add Elixhauser comorbidity features based on ICD codes.
    """
    print("Adding comorbidity features...")

    if diagnoses_df.empty:
        print("Warning: No diagnoses data available for comorbidity calculation")
        features_df['COMORBIDITY_COUNT'] = 0
        return features_df

    icd_column = None
    for col in ['ICD9_CODE', 'ICD_CODE']:
        if col in diagnoses_df.columns:
            icd_column = col
            break

    if icd_column is None:
        print("Warning: No ICD code column found in diagnoses data")
        features_df['COMORBIDITY_COUNT'] = 0
        return features_df

    comorbidities = {
        'CHF':      ['428'],
        'ARRHY':    ['426', '427'],
        'VALVE':    ['394', '395', '396', '397', '424'],
        'PULMCIRC': ['415', '416', '417'],
        'PERIVASC': ['440', '441', '442', '443', '444', '447'],
        'HTN':      ['401', '402', '403', '404', '405'],
        'PARA':     ['342', '343', '344'],
        'NEURO':    ['330', '331', '332', '333', '334', '335', '336', '337'],
        'CHRNLUNG': ['490', '491', '492', '493', '494', '495', '496',
                     '500', '501', '502', '503', '504', '505'],
        'DM':       ['250'],
        'RENLFAIL': ['585', '586', 'V56'],
        'LIVER':    ['570', '571', '572', '573'],
        'ULCER':    ['531', '532', '533', '534'],
        'CANCER':   [str(i) for i in range(140, 210)],
        'DEPRESSION': ['300.4', '301.12', '309.0', '309.1', '311'],
    }

    diag_subset = diagnoses_df[['HADM_ID', icd_column]].copy()
    diag_subset[icd_column] = diag_subset[icd_column].astype(str).str.replace('.', '')

    comorbidity_counts = {c: 0 for c in comorbidities}

    for comorbidity in comorbidities:
        features_df[f'CM_{comorbidity}'] = 0

    for comorbidity, icd_codes in comorbidities.items():
        for icd_code in icd_codes:
            matching = diag_subset[diag_subset[icd_column].str.startswith(icd_code, na=False)]
            if not matching.empty:
                hadm_ids_with = matching['HADM_ID'].unique()
                mask = features_df['HADM_ID'].isin(hadm_ids_with)
                features_df.loc[mask, f'CM_{comorbidity}'] = 1
                comorbidity_counts[comorbidity] += mask.sum()

    comorbidity_cols = [col for col in features_df.columns if col.startswith('CM_')]
    features_df['COMORBIDITY_COUNT'] = features_df[comorbidity_cols].sum(axis=1)

    print("\nComorbidity Statistics:")
    for comorbidity, count in sorted(comorbidity_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / len(features_df)) * 100
            print(f"  {comorbidity}: {count} patients ({percentage:.1f}%)")

    return features_df


def add_note_features(features_df, noteevents_df):
    """
    Extract structured features from discharge summaries using LlamaParse + Claude.

    LlamaParse normalises raw clinical note text before it is sent to Claude
    for field extraction. Features include discharge condition, follow-up
    timeframe, high-risk medications, social support flags, and risk keywords.

    Parameters
    ----------
    features_df   : DataFrame with HADM_ID column
    noteevents_df : MIMIC NOTEEVENTS table

    Returns
    -------
    features_df with ~9 new NOTE_* columns merged in
    """
    if noteevents_df is None or noteevents_df.empty:
        print("Skipping note features — NOTEEVENTS not loaded.")
        return features_df

    try:
        print("\nExtracting discharge note features with LlamaParse + Claude...")

        extractor = NoteFeatureExtractor(
            llamaparse_api_key=os.environ.get("LLAMAPARSE_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            # Set higher for production; 500 is sensible for development runs
            max_notes=500,
            cache_path="note_features_cache.parquet",
        )

        note_features = extractor.run(
            noteevents_df,
            hadm_ids=features_df['HADM_ID'].tolist(),
        )

        features_df = features_df.merge(note_features, on='HADM_ID', how='left')

        # Fill NAs for admissions without a discharge summary
        note_cols = [c for c in note_features.columns if c != 'HADM_ID']
        defaults = {
            'NOTE_MEDICATIONS_COUNT':   0,
            'NOTE_FOLLOWUP_DAYS':       -1,
            'NOTE_FOLLOWUP_SPECIFIED':   0,
            'NOTE_SOCIAL_SUPPORT':       0,
            'NOTE_SUBSTANCE_USE':        0,
            'NOTE_DISCHARGE_CONDITION':  np.nan,
            'NOTE_HIGH_RISK_MED':        0,
            'NOTE_COMORBIDITY_MENTIONS': 0,
            'NOTE_RISK_KEYWORDS':        0,
        }
        for col, default in defaults.items():
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(default)

        note_cols_present = [c for c in note_cols if c in features_df.columns]
        print(f"Note features added: {note_cols_present}")
        print(f"Feature dataset shape after note features: {features_df.shape}")

    except Exception as e:
        print(f"Warning: Note feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without note features.")

    return features_df


def add_external_enrichment(features_df, prescriptions_df, diagnoses_df):
    """
    Enrich features with drug interaction signals and clinical guideline risk
    tiers scraped from DrugBank and ACC/AHA pages via Firecrawl + Claude.

    Drug interactions: flags high-risk polypharmacy combinations from the
    PRESCRIPTIONS table, scored against Firecrawl-scraped DrugBank content.

    Guideline risk tiers: maps primary diagnoses to CMS readmission measure
    categories and assigns evidence-based risk tiers from scraped guidelines.

    Parameters
    ----------
    features_df      : DataFrame with HADM_ID column
    prescriptions_df : MIMIC PRESCRIPTIONS table
    diagnoses_df     : MIMIC DIAGNOSES_ICD table

    Returns
    -------
    features_df with ~8 new drug/guideline columns merged in
    """
    try:
        print("\nEnriching features with Firecrawl (drug interactions + guidelines)...")

        features_df = enrich_features(
            features_df,
            prescriptions_df,
            diagnoses_df,
            firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        print(f"Feature dataset shape after Firecrawl enrichment: {features_df.shape}")

    except Exception as e:
        print(f"Warning: Firecrawl enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without external enrichment features.")

    return features_df


def explore_data(feature_dataset):
    """
    Exploratory data analysis on the feature dataset.
    """
    print("\nExploratory Data Analysis:")
    print("\nFeature dataset summary:")
    print(feature_dataset.describe())

    readmission_counts = feature_dataset['READMISSION_30D'].value_counts()
    print("\nClass distribution:")
    for label, count in readmission_counts.items():
        percentage = (count / len(feature_dataset)) * 100
        print(f"  Readmitted = {label}: {count} ({percentage:.1f}%)")

    if 'GENDER' in feature_dataset.columns:
        gender_counts = feature_dataset['GENDER'].value_counts()
        print("\nGender distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / len(feature_dataset)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=feature_dataset, x='AGE', hue='READMISSION_30D', bins=20, multiple='dodge')
    plt.title('Age Distribution by Readmission Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('plots/age_distribution.png')
    print("Saved age distribution plot")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=feature_dataset, x='LOS_DAYS', hue='READMISSION_30D', bins=20, multiple='dodge')
    plt.title('Length of Stay Distribution by Readmission Status')
    plt.xlabel('Length of Stay (days)')
    plt.ylabel('Count')
    plt.xlim(0, 30)
    plt.savefig('plots/los_distribution.png')
    print("Saved LOS distribution plot")

    numerical_features = feature_dataset.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14, 12))
    correlation = feature_dataset[numerical_features].corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    print("Saved correlation heatmap")

    comorbidity_cols = [col for col in feature_dataset.columns if col.startswith('CM_')]
    if comorbidity_cols:
        plt.figure(figsize=(12, 8))
        cm_counts = feature_dataset[comorbidity_cols].sum().sort_values(ascending=False)
        sns.barplot(x=cm_counts.values, y=cm_counts.index)
        plt.title('Comorbidity Frequency')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('plots/comorbidity_frequency.png')
        print("Saved comorbidity frequency plot")

        comorbidity_by_readmission = feature_dataset.groupby('READMISSION_30D')[comorbidity_cols].mean()
        plt.figure(figsize=(12, 8))
        comorbidity_by_readmission.T.plot(kind='bar')
        plt.title('Comorbidity Frequency by Readmission Status')
        plt.ylabel('Frequency')
        plt.xlabel('Comorbidity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/comorbidity_by_readmission.png')
        print("Saved comorbidity by readmission plot")

    # ── New: Note feature distributions ──────────────────────────────────────
    note_cols = [c for c in feature_dataset.columns if c.startswith('NOTE_')]
    if note_cols:
        plt.figure(figsize=(14, 6))
        note_readmit_means = feature_dataset.groupby('READMISSION_30D')[note_cols].mean()
        note_readmit_means.T.plot(kind='bar')
        plt.title('Note Features by Readmission Status')
        plt.ylabel('Mean Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/note_features_by_readmission.png')
        print("Saved note features by readmission plot")

    # ── New: Drug interaction risk distribution ───────────────────────────────
    if 'HIGH_RISK_INTERACTION_COUNT' in feature_dataset.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=feature_dataset,
            x='READMISSION_30D',
            y='HIGH_RISK_INTERACTION_COUNT',
        )
        plt.title('Drug Interaction Risk by Readmission Status')
        plt.xlabel('Readmitted (0=No, 1=Yes)')
        plt.ylabel('High-Risk Interaction Count')
        plt.savefig('plots/drug_interactions_by_readmission.png')
        print("Saved drug interaction risk plot")

    return feature_dataset


def train_evaluate_models(features_df, use_smote=True):
    """
    Train and evaluate multiple models for readmission prediction.

    Parameters
    ----------
    features_df : DataFrame with target READMISSION_30D
    use_smote   : bool — apply SMOTE oversampling

    Returns
    -------
    (results dict, feature_names)
    """
    print("\nPreparing data for model training...")

    X = features_df.drop(['HADM_ID', 'SUBJECT_ID', 'READMISSION_30D'], axis=1)
    y = features_df['READMISSION_30D']

    # Drop any remaining non-numeric columns (e.g. CMS_CONDITION string)
    X = X.select_dtypes(include=['number', 'bool'])
    X = pd.get_dummies(X, drop_first=True)

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    numerical_cols = X_imputed.select_dtypes(include=['float64', 'int64']).columns
    X_imputed[numerical_cols] = scaler.fit_transform(X_imputed[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.25, random_state=42, stratify=y,
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    print(f"Positive class in training: {sum(y_train)}/{len(y_train)} "
          f"({100*sum(y_train)/len(y_train):.2f}%)")

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE: {len(X_train)} → {len(X_train_resampled)} samples")
            print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
        except ImportError:
            print("Warning: imblearn not installed. Continuing without SMOTE.")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    models = {
        'logistic': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42,
        ),
    }

    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier(
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
        )
    except ImportError:
        print("Warning: XGBoost not installed. Skipping.")

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")

        model.fit(X_train_resampled, y_train_resampled)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        auc_score         = roc_auc_score(y_test, y_pred_proba)
        average_precision = average_precision_score(y_test, y_pred_proba)
        conf_matrix       = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model':         model,
            'auc':           auc_score,
            'avg_precision': average_precision,
            'confusion_matrix': conf_matrix,
            'y_test':        y_test,
            'y_pred':        y_pred,
            'y_pred_proba':  y_pred_proba,
        }

        print(f"AUC: {auc_score:.4f}  |  Avg Precision: {average_precision:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(
            y_test, y_pred_proba,
            name=f"{name} (AUC = {auc_score:.3f})",
            plot_chance_level=True,
        )
        plt.title(f'ROC Curve — {name}')
        plt.savefig(f'plots/roc_curve_{name}.png')

        if name in ['random_forest', 'xgboost'] and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances,
            }).sort_values('Importance', ascending=False)

            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Top 15 Features — {name}')
            plt.tight_layout()
            plt.savefig(f'plots/feature_importance_{name}.png')

    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        PrecisionRecallDisplay.from_predictions(
            result['y_test'], result['y_pred_proba'],
            name=f"{name} (AP = {result['avg_precision']:.3f})",
        )
    plt.title('Precision-Recall Curves')
    plt.savefig('plots/precision_recall_comparison.png')

    best_model_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")

    os.makedirs('models', exist_ok=True)
    joblib.dump(
        results[best_model_name]['model'],
        f'models/readmission_model_{best_model_name}.joblib',
    )
    print(f"Saved best model to models/readmission_model_{best_model_name}.joblib")

    return results, X_train.columns


def predict_readmission_risk(model, new_patient_data, feature_names):
    """
    Predict readmission risk for a new patient.

    Parameters
    ----------
    model            : trained sklearn model
    new_patient_data : dict of feature values
    feature_names    : list of feature names used by the model

    Returns
    -------
    float — readmission probability
    """
    patient_df = pd.DataFrame([new_patient_data])

    for feature in feature_names:
        if feature not in patient_df.columns:
            patient_df[feature] = 0

    patient_df = patient_df[feature_names]
    return model.predict_proba(patient_df)[0, 1]


def main():
    """
    Full pipeline:
      1. Load MIMIC tables (including NOTEEVENTS)
      2. Generate 30-day readmission labels
      3. Build base features (demographics, LOS, diagnoses, labs)
      4. Add Elixhauser comorbidity features
      5. Extract discharge note features via LlamaParse + Claude
      6. Enrich with drug interactions + guideline risk via Firecrawl + Claude
      7. Exploratory data analysis
      8. Train and evaluate models
      9. Example prediction
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    try:
        (admissions, patients, diagnoses, procedures,
         prescriptions, labevents, noteevents) = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # ── 2. Readmission labels ─────────────────────────────────────────────────
    try:
        admissions_with_labels = create_readmission_labels(admissions)
    except Exception as e:
        print(f"Error creating readmission labels: {e}")
        return None

    # ── 3. Base features ──────────────────────────────────────────────────────
    try:
        feature_dataset = create_features(
            admissions_with_labels, patients, diagnoses, procedures, labevents,
        )
    except Exception as e:
        print(f"Error creating features: {e}")
        return None

    # ── 4. Comorbidity features ───────────────────────────────────────────────
    try:
        feature_dataset = add_comorbidity_features(feature_dataset, diagnoses)
    except Exception as e:
        print(f"Error adding comorbidity features: {e}")

    # ── 5. Note features (LlamaParse + Claude) ────────────────────────────────
    feature_dataset = add_note_features(feature_dataset, noteevents)

    # ── 6. External enrichment (Firecrawl + Claude) ───────────────────────────
    feature_dataset = add_external_enrichment(feature_dataset, prescriptions, diagnoses)

    # ── 7. EDA ────────────────────────────────────────────────────────────────
    try:
        explore_data(feature_dataset)
    except Exception as e:
        print(f"Warning: EDA error: {e}")

    # ── 8. Model training ─────────────────────────────────────────────────────
    try:
        model_results, feature_names = train_evaluate_models(feature_dataset)

        print("\nExample prediction with best model:")
        best_model_name = max(model_results, key=lambda k: model_results[k]['auc'])
        best_model      = model_results[best_model_name]['model']

        # Example patient — includes new features from notes and Firecrawl
        example_patient = {
            'AGE':                           65,
            'GENDER_M':                       1,
            'LOS_DAYS':                     5.2,
            'PREV_ADMISSIONS':                2,
            'DIAGNOSIS_COUNT':                8,
            'PROCEDURE_COUNT':                3,
            'COMORBIDITY_COUNT':              4,
            'ADM_TYPE_EMERGENCY':             1,
            'CM_CHF':                         1,
            'CM_HTN':                         1,
            'CM_DM':                          1,
            # Note features
            'NOTE_MEDICATIONS_COUNT':         7,
            'NOTE_FOLLOWUP_DAYS':             7,
            'NOTE_FOLLOWUP_SPECIFIED':        1,
            'NOTE_SOCIAL_SUPPORT':            0,
            'NOTE_SUBSTANCE_USE':             0,
            'NOTE_DISCHARGE_CONDITION':       1,   # Fair
            'NOTE_HIGH_RISK_MED':             1,   # Warfarin on discharge
            'NOTE_COMORBIDITY_MENTIONS':      3,
            'NOTE_RISK_KEYWORDS':             2,
            # Firecrawl features
            'DRUG_COUNT':                    7,
            'HIGH_RISK_INTERACTION_COUNT':   2,
            'HIGH_RISK_DRUG_FLAG':           1,
            'SCRAPED_INTERACTION_RISK_SCORE': 0.4,
            'CMS_NATIONAL_AVG_READMIT_RATE': 21.9,
            'GUIDELINE_RISK_TIER':           3,
            'GUIDELINE_KEY_RISKS_COUNT':     12,
        }

        risk = predict_readmission_risk(best_model, example_patient, feature_names)
        print(f"Example patient's 30-day readmission risk: {risk:.2%}")

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

    print("\nAnalysis complete!")

    return (feature_dataset, admissions, patients, diagnoses,
            procedures, prescriptions, labevents, noteevents)


if __name__ == "__main__":
    result = main()
    if result is not None:
        (feature_dataset, admissions, patients, diagnoses,
         procedures, prescriptions, labevents, noteevents) = result
        print(f"Successfully processed data. Feature dataset: {feature_dataset.shape}")
    else:
        print("Pipeline execution failed. Check error messages above.")

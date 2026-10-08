import json
from functools import lru_cache
from pathlib import Path
import re

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "adr_dataset_combined.csv"
CLASSIFIER_PATH = BASE_DIR / "adr_sbert_classifier.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_sentence_encoder():
    return SentenceTransformer(SENTENCE_MODEL_NAME, device="cpu")


def load_model_metrics():
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


# baseline known side effects for each drug
known_side_effects = {
    "Acetaminophen": ["headache", "liver issues"],
    "Albuterol": ["nausea", "tremor", "nervousness"],
    "Amoxicillin": ["rash", "diarrhea", "nausea"],
    "Aspirin": ["nausea", "stomach pain", "heartburn"],
    "Atorvastatin": ["muscle pain", "liver issues"],
    "Azithromycin": ["diarrhea", "nausea", "abdominal pain"],
    "Baclofen": ["drowsiness", "weakness", "dizziness"],
    "Bisoprolol": ["fatigue", "dizziness", "cold extremities"],
    "Bupropion": ["insomnia", "dry mouth", "headache"],
    "Candesartan": ["dizziness", "fatigue", "low blood pressure"],
    "Carvedilol": ["fatigue", "dizziness", "low blood pressure"],
    "Cephalexin": ["diarrhea", "rash", "nausea"],
    "Ciprofloxacin": ["nausea", "diarrhea", "dizziness"],
    "Citalopram": ["nausea", "dry mouth", "insomnia"],
    "Clonazepam": ["drowsiness", "dizziness", "fatigue"],
    "Doxycycline": ["nausea", "diarrhea", "photosensitivity"],
    "paracetamol": ["rash", "liver damage", "nausea"],
    "Gabapentin": ["drowsiness", "dizziness", "fatigue"],
    "Ibuprofen": ["nausea", "dizziness", "stomach upset"],
    "Metformin": ["diarrhea", "nausea", "abdominal pain"],
    "Sertraline": ["nausea", "insomnia", "dry mouth"],
    "Omeprazole": ["headache", "diarrhea", "nausea"],
    "Pantoprazole": ["diarrhea", "nausea", "headache"],
    "Prednisone": ["weight gain", "insomnia", "mood changes"],
    "Hydrochlorothiazide": ["dizziness", "low potassium", "fatigue"],
    "Losartan": ["dizziness", "fatigue", "low blood pressure"],
    "Lisinopril": ["cough", "dizziness", "fatigue"],
    "Levothyroxine": ["palpitations", "weight loss", "insomnia"],
    "Clopidogrel": ["bleeding", "rash", "diarrhea"],
    "Warfarin": ["bleeding", "nausea", "fatigue"],
    "Insulin": ["hypoglycemia", "weight gain", "injection site reaction"],
    "Furosemide": ["dizziness", "low potassium", "dehydration"],
    "Spironolactone": ["dizziness", "high potassium", "fatigue"],
    "Diazepam": ["drowsiness", "fatigue", "dizziness"],
    "Lorazepam": ["drowsiness", "fatigue", "dizziness"],
    "Alprazolam": ["drowsiness", "fatigue", "dizziness"],
    "Escitalopram": ["nausea", "insomnia", "dry mouth"],
    "Fluoxetine": ["nausea", "insomnia", "headache"],
    "Paroxetine": ["nausea", "insomnia", "dry mouth"],
    "Venlafaxine": ["nausea", "insomnia", "dry mouth"],
    "Duloxetine": ["nausea", "insomnia", "dry mouth"],
    "Tramadol": ["dizziness", "nausea", "constipation"],
    "Morphine": ["drowsiness", "constipation", "nausea"],
    "Oxycodone": ["drowsiness", "constipation", "nausea"],
    "Hydrocodone": ["drowsiness", "constipation", "nausea"],
    "Codeine": ["drowsiness", "constipation", "nausea"],
    "Methotrexate": ["nausea", "fatigue", "mouth sores"],
    "Cyclophosphamide": ["nausea", "hair loss", "fatigue"],
    "Tamoxifen": ["hot flashes", "nausea", "fatigue"],
    "Letrozole": ["hot flashes", "fatigue", "joint pain"],
    "Anastrozole": ["hot flashes", "fatigue", "joint pain"],
    "Cetirizine": ["drowsiness", "dry mouth", "fatigue"],
    "Loratadine": ["headache", "dry mouth", "fatigue"],
    "Fexofenadine": ["headache", "nausea", "fatigue"],
    "Diphenhydramine": ["drowsiness", "dry mouth", "dizziness"],
    "Montelukast": ["headache", "abdominal pain", "fatigue"],
    "Salbutamol": ["tremor", "nervousness", "palpitations"],
    "Budesonide": ["throat irritation", "cough", "headache"],
    "Fluticasone": ["throat irritation", "cough", "headache"],
    "Beclomethasone": ["throat irritation", "cough", "headache"],
    "Tiotropium": ["dry mouth", "cough", "headache"],
    "Ipratropium": ["dry mouth", "cough", "headache"],
    "Ranitidine": ["headache", "diarrhea", "nausea"],
    "Famotidine": ["headache", "diarrhea", "nausea"],
    "Sucralfate": ["constipation", "dry mouth", "nausea"],
    "Metoprolol": ["fatigue", "dizziness", "low blood pressure"],
    "Propranolol": ["fatigue", "dizziness", "low blood pressure"],
    "Amlodipine": ["swelling", "dizziness", "fatigue"],
    "Nifedipine": ["swelling", "dizziness", "headache"],
    "Verapamil": ["constipation", "dizziness", "fatigue"],
    "Diltiazem": ["dizziness", "fatigue", "headache"],
    "Digoxin": ["nausea", "fatigue", "visual disturbances"],
    "Amiodarone": ["nausea", "fatigue", "thyroid issues"],
    "Sotalol": ["fatigue", "dizziness", "low blood pressure"],
    "Erythromycin": ["nausea", "diarrhea", "abdominal pain"],
    "Clarithromycin": ["nausea", "diarrhea", "abdominal pain"],
    "Levofloxacin": ["nausea", "diarrhea", "dizziness"],
    "Moxifloxacin": ["nausea", "diarrhea", "dizziness"],
    "Linezolid": ["nausea", "diarrhea", "headache"],
    "Vancomycin": ["nausea", "diarrhea", "rash"],
    "Gentamicin": ["kidney issues", "ear problems", "nausea"],
    "Tobramycin": ["kidney issues", "ear problems", "nausea"],
    "Streptomycin": ["ear problems", "nausea", "fatigue"],
    "Rifampicin": ["nausea", "liver issues", "rash"],
    "Isoniazid": ["nausea", "liver issues", "fatigue"],
    "Pyrazinamide": ["nausea", "liver issues", "joint pain"],
    "Ethambutol": ["visual disturbances", "nausea", "rash"],
    "Oseltamivir": ["nausea", "vomiting", "headache"],
    "Zidovudine": ["anemia", "nausea", "fatigue"],
    "Lamivudine": ["nausea", "fatigue", "headache"],
    "Efavirenz": ["dizziness", "insomnia", "rash"],
    "Tenofovir": ["nausea", "kidney issues", "fatigue"],
    "Dolutegravir": ["headache", "insomnia", "fatigue"],
    "Raltegravir": ["headache", "insomnia", "fatigue"],
    "Remdesivir": ["nausea", "liver issues", "rash"],
    "Molnupiravir": ["nausea", "diarrhea", "headache"],
    "Paxlovid": ["nausea", "diarrhea", "headache"],
    "Dexamethasone": ["weight gain", "insomnia", "mood changes"],
}

def extract_drug(text):
    text = text.lower()
    detected_drug = None
    for drug in known_side_effects.keys():
        pattern = r"\b" + re.escape(drug.lower()) + r"\b"
        if re.search(pattern, text):
            detected_drug = drug
            break
    return detected_drug

# function to compare reported vs expected side effects
def compare_side_effects(drug, reported_reactions):
    expected = known_side_effects.get(drug, [])
    reported = reported_reactions

    # split into expected vs unexpected
    expected_found = [r for r in reported if r in expected]
    unexpected_found = [r for r in reported if r not in expected]

    return expected_found, unexpected_found

def train_model():
    df = pd.read_csv(DATASET_PATH)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    df = df.dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).str.strip().str.upper()
    if not texts or set(labels) != {"ADR", "NON-ADR"}:
        raise ValueError("Training data must contain both ADR and NON-ADR examples.")

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )
    encoder = get_sentence_encoder()
    X_train = encoder.encode(
        train_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    X_test = encoder.encode(
        test_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, pos_label="ADR", zero_division=0)
        ),
        "recall": float(recall_score(y_test, y_pred, pos_label="ADR", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label="ADR", zero_division=0)),
    }
    if hasattr(classifier, "predict_proba") and len(set(y_test)) == 2:
        adr_index = list(classifier.classes_).index("ADR")
        metrics["roc_auc"] = float(
            roc_auc_score(
                (y_test == "ADR").astype(int),
                classifier.predict_proba(X_test)[:, adr_index],
            )
        )

    joblib.dump(classifier, CLASSIFIER_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Evaluated on {len(y_test)} held-out reviews from {len(texts)} rows.")
    print("Holdout metrics (ADR is the positive class):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved classifier to {CLASSIFIER_PATH}")
    print(f"Saved evaluation metrics to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_model()

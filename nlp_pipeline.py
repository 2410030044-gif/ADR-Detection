import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import random
import csv
import re

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

# Load dataset
df = pd.read_csv("adr_dataset_combined.csv")

# Features and labels
X = df["text"]
y = df["label"]

# Convert text to numeric features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Train a simple classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Try a custom example
example = ["This medicine gave me rashes"]
example_vec = vectorizer.transform(example)
print("Prediction:", model.predict(example_vec)[0])

# Save model and vectorizer
joblib.dump(model, "adr_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved!")

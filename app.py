from flask import Flask, request, render_template
import joblib
from nlp_pipeline import compare_side_effects, known_side_effects
from pymongo import MongoClient
import re

client = MongoClient("mongodb+srv://2410030044-ADR:niha1330@cluster0.xe8kag2.mongodb.net/?retryWrites=true&w=majority")
db = client["adr_database"]          
reviews_collection = db["reviews"]   

model = joblib.load("adr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    prediction = ""
    drug = reaction = time = severity = ""
    expected_effects = unexpected_effects = ""

    if request.method == "POST":
        text = request.form["review"]
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        drug, reaction, time, severity = extract_info(text)

        # compare reported vs expected side effects
        if drug and reaction:
            reactions_list = reaction.split(", ")
            expected_found, unexpected_found = compare_side_effects(drug, reactions_list)
            expected_effects = ", ".join(expected_found) if expected_found else "None"
            unexpected_effects = ", ".join(unexpected_found) if unexpected_found else "None"

        reviews_collection.insert_one({
            "text": text,
            "prediction": prediction,
            "drug": drug if drug else "Unknown",
            "reaction": reaction if reaction else "None",
            "time": time if time else "Not detected",
            "severity": severity if severity else "Not categorized",
            "expected_effects": expected_effects,
            "unexpected_effects": unexpected_effects
        })

    return render_template(
        "detect.html",
        prediction=prediction,
        drug=drug,
        reaction=reaction,
        time=time,
        severity=severity,
        expected_effects=expected_effects,
        unexpected_effects=unexpected_effects
    )

def categorize_severity(reactions_found):
    severity_map = {
        "rash": "mild", "rashes": "mild", "itching": "mild", "hives": "mild",
        "nausea": "moderate", "vomiting": "moderate", "dizzy": "moderate",
        "headache": "moderate", "fatigue": "moderate", "weakness": "moderate",
        "insomnia": "moderate", "diarrhea": "moderate", "constipation": "moderate",
        "anxiety": "moderate", "depression": "moderate",
        "swelling": "severe", "allergic": "severe", "ulcers": "severe",
        "bleeding": "severe", "shortness of breath": "severe",
        "palpitations": "severe", "blurred vision": "severe",
        "weight gain": "severe", "cough": "severe", "fever": "severe"
    }

    severities = []
    for r in reactions_found:
        if r in severity_map:
            severities.append(severity_map[r])

    severities = list(dict.fromkeys(severities))  # remove duplicates
    return ", ".join(severities) if severities else "Not categorized"

def extract_info(text):
    drug = None
    reactions_found = []
    time = None
    drugs = list(known_side_effects.keys())

    for d in drugs:
        pattern = r"\b" + re.escape(d.lower()) + r"\b"
        if re.search(pattern, text.lower()):
            drug = d
            break

    reactions = [
        "rash", "rashes", "nausea", "vomiting", "dizzy", "headache", "allergic",
        "pain", "cramps", "itching", "hives", "swelling", "fatigue", "weakness",
        "insomnia", "diarrhea", "constipation", "blurred vision", "weight gain",
        "anxiety", "depression", "irritation", "ulcers", "bleeding", "cough",
        "shortness of breath", "palpitations", "dry mouth", "fever"
    ]

    for r in reactions:
        if r.lower() in text.lower():
            reactions_found.append(r)

    if "after" in text.lower():
        time = "After some time"
    elif "two weeks" in text.lower():
        time = "After two weeks"

    severity = categorize_severity(reactions_found)

    return drug, ", ".join(reactions_found) if reactions_found else None, time, severity

if __name__ == "__main__":
    app.run(debug=True)

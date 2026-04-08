import random
import csv

# Define drugs and reactions
drugs = ["Paracetamol", "Ibuprofen", "Metformin", "Amoxicillin", "Aspirin",
         "Diclofenac", "Atorvastatin", "Omeprazole", "Prednisone", "Sertraline", "Cetirizine", "Penicillin"]

adr_reactions = [
    "rash", "rashes", "nausea", "vomiting", "dizzy", "headache", "allergic reaction",
    "pain", "cramps", "itching", "hives", "swelling", "fatigue", "weakness",
    "insomnia", "diarrhea", "constipation", "blurred vision", "weight gain",
    "anxiety", "depression", "irritation", "ulcers", "bleeding", "cough",
    "shortness of breath", "palpitations", "dry mouth", "fever"
]

non_adr_phrases = [
    "worked perfectly for my headache",
    "helped reduce fever",
    "controlled my blood sugar well",
    "cleared my throat infection",
    "relieved my acidity issues",
    "helped with chest pain relief",
    "eased my back pain",
    "lowered my cholesterol levels",
    "improved my mood and focus",
    "reduced my inflammation"
]

# Generate synthetic dataset
rows = []
for _ in range(200):  # generate 200 synthetic rows
    drug = random.choice(drugs)
    if random.random() < 0.5:  # ADR case
        reaction = random.choice(adr_reactions)
        sentence = f"{drug} caused {reaction} after use"
        label = "ADR"
    else:  # NON-ADR case
        phrase = random.choice(non_adr_phrases)
        sentence = f"{drug} {phrase}"
        label = "NON-ADR"
    rows.append([drug, sentence, label])

# Save to CSV
with open("synthetic_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drug_name", "text", "label"])
    writer.writerows(rows)

print("Synthetic dataset generated: synthetic_dataset.csv")

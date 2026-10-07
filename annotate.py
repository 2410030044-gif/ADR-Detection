from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["adr_detection"]
collection = db["drug_reviews"]

def annotate_reviews(limit=10):
    reviews = collection.find({"label": None}).limit(limit)

    for review in reviews:
        print("\n--- Review ---")
        print(f"Drug: {review['drug_name']}")
        print(f"Text: {review['text']}")

        label = input("Label this as ADR (adverse) or NON-ADR (neutral)? ").strip().upper()

        if label not in ["ADR", "NON-ADR"]:
            print("Invalid input. Skipping...")
            continue

        collection.update_one({"_id": review["_id"]}, {"$set": {"label": label}})
        print(f"✅ Saved label: {label}")

if __name__ == "__main__":
    print("Starting annotation...")
    annotate_reviews(limit=5)

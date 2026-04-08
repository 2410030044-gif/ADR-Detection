import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["adr_detection"]
collection = db["drug_reviews"]

# Fetch annotated reviews
data = list(collection.find({"label": {"$ne": None}}, {"_id": 0}))

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("adr_dataset.csv", index=False)

print("✅ Exported annotated dataset to adr_dataset.csv")

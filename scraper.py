import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["adr_detection"]
collection = db["drug_reviews"]

def scrape_reviews(drug_name, url, max_reviews=20):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Selector for reviews (site-specific)
    reviews = soup.find_all("p", class_="ddc-comment-content")

    data = []
    for r in reviews[:max_reviews]:
        text = r.get_text(strip=True)
        entry = {
            "drug_name": drug_name,
            "text": text,
            "label": None  # To annotate later
        }
        data.append(entry)
    return data

def save_to_mongo(data):
    if data:
        collection.insert_many(data)
        print(f"Inserted {len(data)} reviews into MongoDB.")
    else:
        print("No data found.")

if __name__ == "__main__":
    drug = "Paracetamol"
    url = "https://www.drugs.com/comments/ibuprofen/"
    reviews_data = scrape_reviews(drug, url, max_reviews=10)
    save_to_mongo(reviews_data)

    # Show a sample
    for doc in collection.find().limit(5):
        print(doc)

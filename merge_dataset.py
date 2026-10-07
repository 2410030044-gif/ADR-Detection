import pandas as pd

# Load both datasets
manual = pd.read_csv("adr_dataset.csv")
synthetic = pd.read_csv("synthetic_dataset.csv")

# Merge them
combined = pd.concat([manual, synthetic], ignore_index=True)

# Save combined dataset
combined.to_csv("adr_dataset_combined.csv", index=False)

print("Combined dataset saved as adr_dataset_combined.csv")


import pandas as pd
from datasets import load_dataset

print("Downloading sentiment dataset...")

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")

df = pd.DataFrame({
    "text": dataset["text"],
    "label": dataset["label"]
})

# Select 50 negative and 50 positive reviews
negative = df[df["label"] == 0].sample(n=50, random_state=42)
positive = df[df["label"] == 1].sample(n=50, random_state=42)

final_data = pd.concat([negative, positive])
final_data = final_data.sample(frac=1, random_state=42)

# Convert numeric labels to words
final_data["label"] = final_data["label"].map({
    0: "negative",
    1: "positive"
})

final_data.to_csv("data/eval_inputs.csv", index=False)

print("\nDataset created successfully!")
print("Total reviews:", len(final_data))
print(final_data["label"].value_counts())
print("\nSaved to data/eval_inputs.csv")

import pandas as pd

data = pd.read_csv("data/eval_inputs.csv")

print(data.head())
print("\nTotal reviews:", len(data))
print("\nLabel counts:")
print(data["label"].value_counts())
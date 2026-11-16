
import os
import pandas as pd
import mlflow

from mlflow.tracking import MlflowClient

# --------------------------------------------------
# 1. MODEL SETTINGS
# --------------------------------------------------

MODEL_NAME = "sentiment-model"
MODEL_ALIAS = "champion"

client = MlflowClient()

# --------------------------------------------------
# 2. GET CHAMPION VERSION AND MODEL ID
# --------------------------------------------------

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

version_number = model_version.version

hf_model_id = model_version.tags.get(
    "hf_model_id",
    "Not available"
)

# --------------------------------------------------
# 3. LOAD MODEL THROUGH MLFLOW
# --------------------------------------------------

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(model_uri)

# --------------------------------------------------
# 4. LOAD NEW, UNSEEN DATA
# --------------------------------------------------

data = pd.read_csv("data/new_data.csv")

data.columns = data.columns.str.strip()

if "text" not in data.columns:
    raise ValueError(
        f"'text' column not found. Available columns: {list(data.columns)}"
    )

data["text"] = (
    data["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

data = data[data["text"] != ""].copy()

if data.empty:
    raise ValueError("No valid text examples found.")

# --------------------------------------------------
# 5. PREDICT USING MLFLOW
# --------------------------------------------------

predictions = model.predict(data[["text"]])

results = pd.DataFrame({
    "text": predictions["text"],
    "predicted": predictions["predicted"]
})

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

output_file = "outputs/new_data_predictions.csv"

results.to_csv(output_file, index=False)

# --------------------------------------------------
# 7. DISPLAY FINAL OUTPUT
# --------------------------------------------------

print("\nCHAMPION MODEL INFORMATION")
print("Registered model:", MODEL_NAME)
print("Model alias:", MODEL_ALIAS)
print("Model version:", version_number)
print("Hugging Face model ID:", hf_model_id)

print("\nNEW DATA PREDICTIONS")
print(results.to_string(index=False))

print("\nTotal new examples:", len(results))
print("Saved to:", output_file)
print("\nPrediction completed successfully!")

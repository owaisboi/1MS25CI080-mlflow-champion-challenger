
import os
import time
import json
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import transformers
import torch
import sklearn

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

STUDENT_USN = "1MS25CI080"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
PRIMARY_METRIC = "macro_f1"
MINIMUM_MARGIN = 0.01

os.makedirs("outputs/plots", exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda" if DEVICE == 0 else "cpu"

# --------------------------------------------------
# 2. LOAD EVALUATION DATA
# --------------------------------------------------

data = pd.read_csv("data/eval_inputs.csv")

data["text"] = data["text"].fillna("").astype(str).str.strip()
data["label"] = (
    data["label"].fillna("").astype(str).str.lower().str.strip()
)

data = data[
    data["label"].isin(["positive", "negative"])
].copy()

if len(data) == 0:
    raise ValueError("No valid labelled evaluation examples found.")

texts = data["text"].tolist()
actual = data["label"].tolist()

data.to_csv("outputs/eval_inputs.csv", index=False)

print("Evaluation samples:", len(data))
print("Device:", DEVICE_NAME)

# --------------------------------------------------
# 3. MLFLOW SETUP
# --------------------------------------------------

mlflow.set_experiment("Champion-Challenger")

models = [
    {
        "name": "DistilBERT",
        "run_name": "A_sentiment_distilbert-sst2",
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate",
        "artifact_name": "distilbert_model"
    },
    {
        "name": "RoBERTa",
        "run_name": "B_sentiment_roberta-large",
        "model_id": "siebert/sentiment-roberta-large-english",
        "role": "challenger_candidate",
        "artifact_name": "roberta_model"
    }
]

# --------------------------------------------------
# 4. MLFLOW MODEL WRAPPER
# --------------------------------------------------

class SentimentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            device=DEVICE
        )

    def predict(self, context, model_input, params=None):

        if isinstance(model_input, pd.DataFrame):
            input_data = model_input.copy()
        else:
            input_data = pd.DataFrame(model_input)

        input_texts = (
            input_data["text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        input_texts = [
            text if text else " " for text in input_texts
        ]

        predictions = self.classifier(
            input_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE
        )

        return pd.DataFrame({
            "text": input_texts,
            "predicted": [
                p["label"].lower() for p in predictions
            ]
        })

# --------------------------------------------------
# 5. EVALUATE BOTH MODELS
# --------------------------------------------------

results = []
all_predictions = {}

for model_info in models:

    name = model_info["name"]
    model_id = model_info["model_id"]

    print(f"\nEvaluating {name}...")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=DEVICE
    )

    start = time.perf_counter()

    predictions = classifier(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE
    )

    total_time = time.perf_counter() - start

    predicted = [
        p["label"].lower() for p in predictions
    ]

    avg_latency_ms = (total_time / len(texts)) * 1000

    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted,
            pos_label="positive",
            zero_division=0
        ),
        "recall": recall_score(
            actual, predicted,
            pos_label="positive",
            zero_division=0
        ),
        "macro_f1": f1_score(
            actual, predicted,
            average="macro",
            zero_division=0
        ),
        "avg_latency_ms": avg_latency_ms,
        "total_inference_s": total_time
    }

    output = pd.DataFrame({
        "text": texts,
        "actual": actual,
        "predicted": predicted,
        "correct": [
            a == p for a, p in zip(actual, predicted)
        ]
    })

    prediction_file = (
        f"outputs/{name.lower()}_predictions.csv"
    )
    output.to_csv(prediction_file, index=False)

    all_predictions[name] = predicted

    # Confusion matrix
    cm = confusion_matrix(
        actual,
        predicted,
        labels=["negative", "positive"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    plot_file = (
        f"outputs/plots/{name.lower()}_confusion_matrix.png"
    )

    plt.savefig(plot_file)
    plt.close()

    # MLflow model run
    with mlflow.start_run(run_name=model_info["run_name"]):

        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment_classification",
            "role": model_info["role"],
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "eval_samples": len(data),
            "seed": SEED,
            "device": DEVICE_NAME
        })

        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "dataset_name": "rotten_tomatoes",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "sklearn_version": sklearn.__version__
        })

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(prediction_file)
        mlflow.log_artifact("outputs/eval_inputs.csv")
        mlflow.log_artifact(plot_file)

        mlflow.pyfunc.log_model(
            name=model_info["artifact_name"],
            python_model=SentimentModel(model_id)
        )

    results.append({
        "model": name,
        "hf_model_id": model_id,
        **metrics
    })

    print(name, metrics)

# --------------------------------------------------
# 6. SAVE COMPARISON RESULTS
# --------------------------------------------------

summary = pd.DataFrame(results)

summary.to_csv(
    "outputs/metrics_summary.csv",
    index=False
)

with open("outputs/metrics_summary.json", "w") as file:
    json.dump(results, file, indent=4)

# --------------------------------------------------
# 7. CHAMPION DECISION RULE
# --------------------------------------------------

distilbert_result = next(
    r for r in results if r["model"] == "DistilBERT"
)

roberta_result = next(
    r for r in results if r["model"] == "RoBERTa"
)

delta = (
    roberta_result[PRIMARY_METRIC]
    - distilbert_result[PRIMARY_METRIC]
)

latency_ratio = (
    roberta_result["avg_latency_ms"]
    / max(distilbert_result["avg_latency_ms"], 0.000001)
)

if delta >= MINIMUM_MARGIN and latency_ratio <= 5:
    winner = "RoBERTa"
else:
    winner = "DistilBERT"

challenger_improves = winner == "RoBERTa"

print("\nCHAMPION DECISION")
print("Primary metric:", PRIMARY_METRIC)
print("Challenger improvement:", round(delta, 4))
print("Latency ratio:", round(latency_ratio, 2))
print("Selected Champion:", winner)

# --------------------------------------------------
# 8. MODEL COMPARISON PLOT
# --------------------------------------------------

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "macro_f1"
]

summary.set_index("model")[metric_columns].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Champion-Challenger Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()

comparison_plot = "outputs/plots/model_comparison.png"

plt.savefig(comparison_plot)
plt.close()

# --------------------------------------------------
# 9. MODEL DISAGREEMENTS
# --------------------------------------------------

disagreements = pd.DataFrame({
    "text": texts,
    "actual": actual,
    "distilbert_prediction": all_predictions["DistilBERT"],
    "roberta_prediction": all_predictions["RoBERTa"]
})

disagreements = disagreements[
    disagreements["distilbert_prediction"]
    != disagreements["roberta_prediction"]
].head(5)

disagreements["comment"] = (
    "The two models predicted different sentiments."
)

disagreements.to_csv(
    "outputs/model_disagreements.csv",
    index=False
)

# --------------------------------------------------
# 10. LOG SUMMARY RUN
# --------------------------------------------------

with mlflow.start_run(run_name="compare_sentiment"):

    mlflow.log_params({
        "evaluation_samples": len(data),
        "primary_metric": PRIMARY_METRIC,
        "minimum_margin": MINIMUM_MARGIN,
        "tie_breaker": "Retain DistilBERT unless RoBERTa improves by 0.01 and latency is within 5x",
        "seed": SEED
    })

    mlflow.set_tag("winner", winner)
    mlflow.set_tag("student_usn", STUDENT_USN)

    mlflow.log_metric("delta_primary_metric", delta)
    mlflow.log_metric("latency_ratio", latency_ratio)

    mlflow.log_artifact("outputs/metrics_summary.csv")
    mlflow.log_artifact("outputs/metrics_summary.json")
    mlflow.log_artifact("outputs/eval_inputs.csv")
    mlflow.log_artifact("outputs/model_disagreements.csv")
    mlflow.log_artifact(comparison_plot)

    mlflow.log_artifacts(
        "outputs/plots",
        artifact_path="plots"
    )

    if os.path.exists("README.md"):
        mlflow.log_artifact("README.md")

    if os.path.exists("requirements.txt"):
        mlflow.log_artifact("requirements.txt")

# --------------------------------------------------
# 11. FINAL RESULTS
# --------------------------------------------------

print("\nFINAL MODEL COMPARISON")
print(summary.to_string(index=False))

print("\nEvaluation samples:", len(data))
print("Selected Champion:", winner)
print("Plots saved in outputs/plots/")
print("JSON summary saved in outputs/metrics_summary.json")
print("Model disagreements saved successfully.")
print("\nEvaluation completed successfully!")

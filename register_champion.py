
import time
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Champion-Challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

client = MlflowClient()

# --------------------------------------------------
# 1. FIND THE EXPERIMENT
# --------------------------------------------------

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise Exception("Experiment not found. Run evaluate.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"]
)

# --------------------------------------------------
# 2. FIND THE LATEST MODEL RUNS
# --------------------------------------------------

required_runs = {
    "A_sentiment_distilbert-sst2": None,
    "B_sentiment_roberta-large": None
}

comparison_run = None

for run in runs:
    run_name = run.data.tags.get("mlflow.runName")

    if run_name in required_runs and required_runs[run_name] is None:
        required_runs[run_name] = run

    if run_name == "compare_sentiment" and comparison_run is None:
        comparison_run = run

if any(run is None for run in required_runs.values()):
    raise Exception("Could not find both model runs. Run evaluate.py first.")

if comparison_run is None:
    raise Exception("Comparison run not found. Run evaluate.py first.")

# --------------------------------------------------
# 3. READ THE WINNER FROM THE COMPARISON RUN
# --------------------------------------------------

winner = comparison_run.data.tags.get("winner")

if winner not in ["DistilBERT", "RoBERTa"]:
    raise Exception("Invalid Champion decision in comparison run.")

print("Selected Champion:", winner)

# --------------------------------------------------
# 4. REGISTER BOTH MODELS
# --------------------------------------------------

model_details = {
    "DistilBERT": {
        "run": required_runs["A_sentiment_distilbert-sst2"],
        "artifact": "distilbert_model"
    },
    "RoBERTa": {
        "run": required_runs["B_sentiment_roberta-large"],
        "artifact": "roberta_model"
    }
}

versions = {}

for name, details in model_details.items():

    run = details["run"]
    model_uri = f"runs:/{run.info.run_id}/{details['artifact']}"

    print(f"\nRegistering {name}...")

    version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    versions[name] = version.version

    hf_model_id = run.data.params["hf_model_id"]

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version.version,
        "hf_model_id",
        hf_model_id
    )

# --------------------------------------------------
# 5. WAIT FOR REGISTRATION
# --------------------------------------------------

for version_number in versions.values():

    for attempt in range(120):

        model_version = client.get_model_version(
            REGISTERED_MODEL_NAME,
            version_number
        )

        if model_version.status == "READY":
            break

        if model_version.status == "FAILED":
            raise Exception(
                f"Model version {version_number} registration failed."
            )

        time.sleep(2)

    else:
        raise TimeoutError(
            f"Model version {version_number} did not become READY."
        )

# --------------------------------------------------
# 6. ASSIGN CHAMPION AND CHALLENGER ALIASES
# --------------------------------------------------

challenger = (
    "RoBERTa" if winner == "DistilBERT" else "DistilBERT"
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    versions[winner]
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    versions[challenger]
)

# --------------------------------------------------
# 7. DISPLAY FINAL RESULT
# --------------------------------------------------

print("\nREGISTRATION COMPLETED SUCCESSFULLY!")

print(
    f"Champion: {winner}, version {versions[winner]}"
)

print(
    f"Challenger: {challenger}, version {versions[challenger]}"
)

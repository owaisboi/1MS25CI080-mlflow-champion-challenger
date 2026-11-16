# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and Models

**Task:** Sentiment Classification

This project compares two pretrained Hugging Face sentiment classification models using the same labelled evaluation dataset. Both models predict whether a given text expresses positive or negative sentiment.

**Model A – Champion Candidate:**
- Model: DistilBERT
- Hugging Face ID: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

**Model B – Challenger Candidate:**
- Model: RoBERTa
- Hugging Face ID: `siebert/sentiment-roberta-large-english`

**Evaluation dataset:** `data/eval_inputs.csv`  
**Dataset source:** Rotten Tomatoes  
**Evaluation samples:** 100 (50 positive and 50 negative)  
**New prediction dataset:** `data/new_data.csv` (6 examples)

Both models are evaluated on identical inputs. Their predictions, performance metrics, inference times and model artifacts are recorded using MLflow.

## 2. How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare the evaluation dataset

```bash
python data/prepare_dataset.py
```

This prepares 100 labelled reviews, containing 50 positive and 50 negative examples.

### Step 3: Run model evaluation

```bash
python evaluate.py
```

This evaluates both models, calculates performance metrics, logs model runs in MLflow and saves prediction files and evaluation plots.

### Step 4: Register models and assign aliases

```bash
python register_champion.py
```

This registers both models under the registered model name `sentiment-model` and assigns the `champion` and `challenger` aliases according to the evaluation decision.

### Step 5: Start the MLflow UI

```bash
mlflow ui
```

Open the following address in your browser:

`http://127.0.0.1:5000`


```

The script loads the Champion using its MLflow registry alias and predicts sentiment on six new examples. The predictions are saved to `outputs/new_data_predictions.csv`.

## 3. Champion–Challenger Decision Rule

The following decision rule was documented before the final evaluation:

- **Primary metric:** Macro-F1 score. Higher is better.
- **Minimum improvement margin:** The Challenger must improve macro-F1 by at least 0.01 (one percentage point) to replace the Champion.
- **Tie-breaker:** If the improvement is below 0.01, the faster model is preferred.
- **Latency guardrail:** A model taking more than five times the Champion's inference time will not be selected.

Model A (DistilBERT) is initially treated as the Champion, while Model B (RoBERTa) is the Challenger.

## 4. Results Table

Results from the final evaluation on 100 labelled Rotten Tomatoes reviews:

| Metric | DistilBERT | RoBERTa | Difference (RoBERTa − DistilBERT) |
|---|---:|---:|---:|
| Accuracy | 0.9100 | 0.9400 | +0.0300 |
| Precision | 0.9362 | 0.9583 | +0.0222 |
| Recall | 0.8800 | 0.9200 | +0.0400 |
| Macro-F1 | 0.9099 | 0.9400 | +0.0301 |
| Average latency (ms) | 13.5189 | 96.8339 | +83.3150 |
| Total inference time (seconds) | 1.3519 | 9.6834 | +8.3315 |

## 5. Decision and Conclusion

The final evaluation compared DistilBERT and RoBERTa on the same 100 labelled Rotten Tomatoes reviews. RoBERTa achieved an accuracy of 0.94, compared with 0.91 for DistilBERT. It also achieved higher precision (0.9583 compared with 0.9362), recall (0.92 compared with 0.88), and macro-F1 (0.9400 compared with 0.9099). Its macro-F1 improvement was approximately 0.0301, exceeding the minimum required improvement margin of 0.01.

However, the decision rule also includes a latency guardrail. DistilBERT recorded an average inference latency of approximately 13.52 ms, whereas RoBERTa recorded approximately 96.83 ms. RoBERTa was therefore about 7.16 times slower, exceeding the permitted five-times latency limit.



## 6. Limitations and Declaration

### Limitations

- The evaluation uses 100 labelled examples, which may not represent all types of sentiment.
- Results may vary on larger or more diverse datasets.
- Inference latency depends on hardware and system load.
- Both models were evaluated using a single dataset.
- A larger evaluation dataset would provide stronger evidence of generalisation.

### Declaration

This project was developed individually as part of the Skill Lab assignment on MLOps with MLflow.

ChatGPT was used as an AI assistant for guidance, code explanations, debugging assistance and documentation support. The implementation and results were tested in the local development environment.

```
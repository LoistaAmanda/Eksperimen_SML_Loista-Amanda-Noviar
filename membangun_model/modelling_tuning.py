from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn


# ===============================
# PATH
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# MLFLOW
# ===============================
mlflow.set_experiment("IMDB_Tuning_Experiment")


# ===============================
# PIPELINE (WAJIB)
# ===============================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    ))
])


# ===============================
# TRAINING & LOGGING
# ===============================
with mlflow.start_run():

    # ✅ INI YANG BENAR
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.log_param("model", "RandomForest + TFIDF")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 25)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(str(cm_path))

    # Classification Report
    report_path = ARTIFACT_DIR / "classification_report_tuning.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))

    mlflow.log_artifact(str(report_path))

    # Metric JSON
    metric_info = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1
    }

    metric_path = ARTIFACT_DIR / "metric_info.json"
    with open(metric_path, "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact(str(metric_path))

    # LOG MODEL
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model"
    )

    print("MODEL TUNING BERHASIL")

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

# MLflow
import mlflow
import mlflow.sklearn


# =========================================================
# PATH CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# SET MLFLOW EXPERIMENT (LOCAL)
# =========================================================
mlflow.set_experiment("IMDB_Tuning_Experiment")


# =========================================================
# PIPELINE + HYPERPARAMETER TUNING
# =========================================================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=200,      # TUNING
        max_depth=25,          # TUNING
        random_state=42,
        n_jobs=-1
    ))
])


# =========================================================
# TRAINING & MANUAL LOGGING
# =========================================================
with mlflow.start_run():

    # -------------------------
    # Training
    # -------------------------
    pipeline.fit(X_train, y_train)

    # -------------------------
    # Prediction
    # -------------------------
    y_pred = pipeline.predict(X_test)

    # -------------------------
    # Metrics (SETARA AUTOLOG)
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_weighted", f1)

    # -------------------------
    # Log Parameters
    # -------------------------
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 25)
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,2)")

    # =====================================================
    # CONFUSION MATRIX (ARTIFACT)
    # =====================================================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(y.unique()),
        yticklabels=sorted(y.unique())
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Training Confusion Matrix")

    cm_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(str(cm_path))

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================
    report = classification_report(y_test, y_pred)

    report_path = ARTIFACT_DIR / "classification_report_tuning.txt"
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(str(report_path))

    # =====================================================
    # METRIC INFO JSON
    # =====================================================
    metric_info = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1
    }

    metric_info_path = ARTIFACT_DIR / "metric_info.json"
    with open(metric_info_path, "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact(str(metric_info_path))

    # =====================================================
    # LOG MODEL 
    # =====================================================
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model"
    )

    print("Model tuning & artefak berhasil dilog ke MLflow")

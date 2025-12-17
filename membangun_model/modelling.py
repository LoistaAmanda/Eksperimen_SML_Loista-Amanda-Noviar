from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# =========================
# Paths
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"
ARTIFAK_DIR = ROOT_DIR / "artifacts"
ARTIFAK_DIR.mkdir(exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MLflow Autologging
# =========================
mlflow.sklearn.autolog()

# =========================
# Pipeline
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# =========================
# Train model
# =========================
with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)

    # =========================
    # Predictions & Metrics
    # =========================
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics manually (advanced)
    mlflow.log_metrics({
        "precision_low": report["low"]["precision"],
        "recall_low": report["low"]["recall"],
        "f1_low": report["low"]["f1-score"],
        "precision_medium": report["medium"]["precision"],
        "recall_medium": report["medium"]["recall"],
        "f1_medium": report["medium"]["f1-score"],
        "precision_high": report["high"]["precision"],
        "recall_high": report["high"]["recall"],
        "f1_high": report["high"]["f1-score"],
    })

    # Save confusion matrix as artifact
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["low","medium","high"],
                yticklabels=["low","medium","high"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = ARTIFAK_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(str(cm_path))

    # Save classification report as text
    report_path = ARTIFAK_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(str(report_path))

  

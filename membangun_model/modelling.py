# ============================================
# MODELLING IMDB MOVIE RATING CLASSIFICATION
# ============================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import mlflow.sklearn

# ============================================
# 1. Konfigurasi Path Folder
# ============================================

# Folder utama modelling
ROOT_DIR = Path(__file__).resolve().parent

# Path dataset hasil preprocessing
DATA_PATH = ROOT_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"

# Folder untuk menyimpan artefak (gambar, laporan)
ARTIFAK_DIR = ROOT_DIR / "artifacts"
ARTIFAK_DIR.mkdir(exist_ok=True)

# ============================================
# 2. Load Dataset
# ============================================

df = pd.read_csv(DATA_PATH)

# Fitur (teks) dan target (label kelas rating)
X = df["clean_text"]
y = df["label"]

# Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# 3. Pipeline Model NLP
# ============================================

# Pipeline:
# - TF-IDF untuk ekstraksi fitur teks
# - Random Forest untuk klasifikasi multi-kelas
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# ============================================
# 4. Training + MLflow Manual Logging
# ============================================

with mlflow.start_run():

    # =========================
    # Training Model
    # =========================
    pipeline.fit(X_train, y_train)

    # =========================
    # Logging Parameter Model
    # (WAJIB untuk ADVANCE)
    # =========================
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("test_size", 0.2)

    # =========================
    # Prediksi dan Evaluasi
    # =========================
    y_pred = pipeline.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # =========================
    # Logging Metrics (Manual)
    # =========================
    mlflow.log_metric("accuracy", report["accuracy"])

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

    # =========================
    # Logging Model (WAJIB ADVANCE)
    # =========================
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model"
    )

    # =========================
    # Simpan & Log Confusion Matrix
    # =========================
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["low", "medium", "high"],
        yticklabels=["low", "medium", "high"]
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix")

    cm_path = ARTIFAK_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(str(cm_path))

    # =========================
    # Simpan & Log Classification Report
    # =========================
    report_path = ARTIFAK_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))

    mlflow.log_artifact(str(report_path))

print("Training dan logging MLflow selesai.")

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import mlflow
import mlflow.sklearn
import dagshub

# =========================
# Inisialisasi DagsHub
# =========================
dagshub.init(
    repo_owner="LoistaAmanda",
    repo_name="Eksperimen_SML_Loista-Amanda-Noviar",
    mlflow=True
)

# =========================
# Path
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"

# =========================
# Load Data
# =========================
df = pd.read_csv(DATA_PATH)
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Aktifkan autolog MLflow
# =========================
mlflow.sklearn.autolog()

# =========================
# Daftar Parameter Tuning
# =========================
param_grid = [
    {"n_estimators": 50, "max_features": 3000},
    {"n_estimators": 100, "max_features": 5000},
    {"n_estimators": 200, "max_features": 8000},
]

# =========================
# Loop Eksperimen
# =========================
for params in param_grid:
    with mlflow.start_run():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=params["max_features"])),
            ("clf", RandomForestClassifier(
                n_estimators=params["n_estimators"],
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metric manual 
        mlflow.log_metric("f1_weighted", f1)

        print(f"Run selesai → {params}, F1 = {f1:.4f}")

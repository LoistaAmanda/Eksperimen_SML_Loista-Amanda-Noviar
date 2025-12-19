import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =========================
# AUTOLOG 
# =========================
mlflow.autolog()

# =========================
# Load Data
# =========================
df = pd.read_csv("dataset_preprocessing/imbd_preprocessed.csv")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Model 
# =========================
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=100))
])

# =========================
# Train
# =========================
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

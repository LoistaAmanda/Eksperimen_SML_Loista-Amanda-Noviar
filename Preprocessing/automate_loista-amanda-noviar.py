from pathlib import Path
import pandas as pd
import re
import numpy as np


# =========================
# Helper Functions
# =========================

def rating_to_label(r):
    if r < 6:
        return "low"
    elif r < 8:
        return "medium"
    else:
        return "high"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# Main Preprocessing Logic
# =========================

def preprocess_data(input_path: Path, output_path: Path):
    # Load dataset
    df = pd.read_csv(input_path)

    # =========================
    # Handle Missing & Cleaning Numerik
    # =========================

    # Year
    df["year"] = (
        df["year"]
        .astype(str)
        .str.extract(r"(\d{4})")
        .astype(float)
    )
    df["year"] = df["year"].fillna(df["year"].median())

    # Votes
    df["votes"] = (
        df["votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["votes"] = df["votes"].fillna(df["votes"].median())

    # Duration
    df["duration"] = (
        df["duration"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    df["duration"] = df["duration"].fillna(df["duration"].median())

    # =========================
    # Handle Missing Kategorikal
    # =========================

    df["certificate"] = df["certificate"].fillna("Unknown")
    df["genre"] = df["genre"].fillna("Unknown")

    # =========================
    # Drop baris penting
    # =========================

    df = df.dropna(subset=["description", "rating"])

    # =========================
    # Labeling
    # =========================

    df["label"] = df["rating"].apply(rating_to_label)

    # =========================
    # Text Cleaning
    # =========================

    df["clean_text"] = df["description"].apply(clean_text)

    # =========================
    # Final Dataset
    # =========================

    processed_df = df[["clean_text", "label"]]

    # Pastikan folder output ada
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    processed_df.to_csv(output_path, index=False)

    print(f"Preprocessing selesai")
    print(f"Output: {output_path}")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent
    PREPROCESSING_DIR = Path(__file__).resolve().parent

    input_csv = ROOT_DIR / "dataset_raw" / "IMBD.csv"

    output_dir = PREPROCESSING_DIR / "dataset_preprocessing"
    output_csv = output_dir / "imbd_preprocessed.csv"

    preprocess_data(input_csv, output_csv)

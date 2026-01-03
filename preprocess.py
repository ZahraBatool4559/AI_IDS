import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data(filename):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_folder, "data", filename)

    print("Loading:", data_path)
    df = pd.read_csv(data_path)
    return df



def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    return df


def encode_labels(df):
    # UNSW-NB15 uses 'Label' column
    df["label"] = df["label"].astype(int)
    return df


def select_features(df):
    # Drop non-numeric and non-useful columns
    drop_cols = ["attack_cat", "label"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["label"]

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    return X, y


# ---------- TEST ----------
if __name__ == "__main__":
    df = load_data("UNSW_NB15_training-set.csv")

    print("Original shape:", df.shape)

    df = clean_data(df)
    print("After cleaning:", df.shape)

    df = encode_labels(df)

    X, y = select_features(df)
    print("Features:", X.shape)
    print("Labels:", y.value_counts())


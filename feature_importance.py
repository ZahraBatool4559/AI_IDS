import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

def load_model():
    return joblib.load("ids_model.pkl")

def load_data(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "data", filename)
    return pd.read_csv(data_path)

if __name__ == "__main__":
    model = load_model()
    df = load_data("UNSW_NB15_training-set.csv")

    # Keep only numeric features used for training
    X = df.drop(columns=["Label", "attack_cat"], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64"])

    # Check lengths
    importances = model.feature_importances_
    features = X.columns
    print("Features length:", len(features))
    print("Importances length:", len(importances))

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    # Plot
    plt.figure(figsize=(10,6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Important Features")
    plt.xlabel("Importance Score")
    plt.show()

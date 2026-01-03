import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# Load dataset
def load_data(filename):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_folder, "data", filename)
    return pd.read_csv(data_path)


# Prepare features and labels
def prepare_data(df):
    X = df.drop(columns=["Label", "attack_cat"], errors="ignore")
    y = df["label"]

    # Keep only numeric data
    X = X.select_dtypes(include=["int64", "float64"])
    return X, y


# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# Main execution
if __name__ == "__main__":
    print("🔹 Loading training data...")
    train_df = load_data("UNSW_NB15_training-set.csv")

    print("🔹 Loading testing data...")
    test_df = load_data("UNSW_NB15_testing-set.csv")

    print("🔹 Preparing data...")
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)

    print("🔹 Training AI model...")
    model = train_model(X_train, y_train)

    print("🔹 Testing model...")
    predictions = model.predict(X_test)

    print("\n✅ Accuracy:", accuracy_score(y_test, predictions))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\n📄 Classification Report:\n", classification_report(y_test, predictions))

    print("💾 Saving model...")
    joblib.dump(model, "ids_model.pkl")

    print("\n🎉 Training complete! Model saved as ids_model.pkl")

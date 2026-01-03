import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def load_data(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", filename)
    return pd.read_csv(path)


def prepare(df):
    X = df.drop(columns=["label", "attack_cat"], errors="ignore")
    y = df["label"]
    X = X.select_dtypes(include=["int64", "float64"])
    return X, y


if __name__ == "__main__":
    train_df = load_data("UNSW_NB15_training-set.csv")
    test_df = load_data("UNSW_NB15_testing-set.csv")

    X_train, y_train = prepare(train_df)
    X_test, y_test = prepare(test_df)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("XGBoost Accuracy:", accuracy_score(y_test, preds))

    joblib.dump(model, "ids_xgboost.pkl")

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def execute_pipeline():
    print("[MLOps Pipeline] Starting pipeline execution...")

    # Gather data payloads
    raw_data = load_iris(as_frame=True)
    df = raw_data.frame

    X = df.iloc[:, :-1]
    y = raw_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train structured model artifacts
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    # Evaluate model
    score = classifier.score(X_test, y_test)

    # Persist model
    os.makedirs("models", exist_ok=True)

    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    print(
        f"[MLOps Pipeline] Target model successfully cached. "
        f"Test Score: {score:.4f}"
    )


if __name__ == "__main__":
    execute_pipeline()
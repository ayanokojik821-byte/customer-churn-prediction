import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing import (
    load_data,
    split_features_target,
    get_feature_types,
    build_preprocessor,
    train_test_split_data,
)


def main():
    # Load data
    df = load_data("data/raw/telco_churn.csv")
    X, y = split_features_target(df)

    # Feature types
    numerical_features, categorical_features = get_feature_types(X)

    # Preprocessor
    preprocessor = build_preprocessor(
        numerical_features, categorical_features
    )

    # Final model (chosen after comparison)
    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    # Train on FULL dataset
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, "models/churn_model.pkl")

    print("Model saved successfully to models/churn_model.pkl")


if __name__ == "__main__":
    main()


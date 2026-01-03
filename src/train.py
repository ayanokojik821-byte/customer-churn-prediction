import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocessing import (
    load_data,
    split_features_target,
    get_feature_types,
    build_preprocessor,
    train_test_split_data,
)


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train model pipeline and print evaluation metrics
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


def main():
    # Load data
    df = load_data("data/raw/telco_churn.csv")

    # Split features and target
    X, y = split_features_target(df)

    # Feature types
    numerical_features, categorical_features = get_feature_types(X)

    # Preprocessor
    preprocessor = build_preprocessor(
        numerical_features, categorical_features
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    # Train and evaluate each model
    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", clf),
            ]
        )
        train_and_evaluate(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            name,
        )


if __name__ == "__main__":
    main()

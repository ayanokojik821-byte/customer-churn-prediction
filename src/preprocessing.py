import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """
    Load raw churn dataset
    """
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """
    Separate features and target
    """
    X = df.drop(columns=[target])
    y = df[target].map({"Yes": 1, "No": 0})
    return X, y


def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical columns
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numerical_features, categorical_features


def build_preprocessor(numerical_features, categorical_features):
    """
    Build ColumnTransformer for preprocessing
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

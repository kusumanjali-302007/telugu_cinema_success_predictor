from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["budget_crore", "runtime_minutes", "release_month"]
CATEGORICAL = ["genre", "director_profile", "cast_profile", "music_profile", "production_profile"]
FEATURE_COLUMNS = NUMERIC + CATEGORICAL
TARGET_CLASS = "success_category"
TARGET_REG = "gross_crore"

def load_dataset(path):
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_CLASS, TARGET_REG, "release_year"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df.drop_duplicates().copy()
    for col in NUMERIC + [TARGET_REG, "release_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_CLASS, TARGET_REG])
    df = df[(df["budget_crore"] >= 0) & (df["gross_crore"] >= 0)]
    if len(df) < 20:
        raise ValueError("At least 20 valid rows are required to train this demo.")
    return df.reset_index(drop=True)

def _preprocessor():
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

def train_models(df, model_dir=None):
    X = df[FEATURE_COLUMNS]
    y_class = df[TARGET_CLASS].astype(str)
    y_reg = df[TARGET_REG].astype(float)
    stratify = y_class if y_class.value_counts().min() >= 2 else None
    X_train, X_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.25, random_state=42, stratify=stratify
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    clf = Pipeline([
        ("prep", _preprocessor()),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")),
    ])
    reg = Pipeline([
        ("prep", _preprocessor()),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42)),
    ])
    clf.fit(X_train, yc_train)
    reg.fit(Xr_train, yr_train)
    class_pred = clf.predict(X_test)
    reg_pred = reg.predict(Xr_test)
    metrics = {
        "classification_accuracy": float(accuracy_score(yc_test, class_pred)),
        "regression_mae_crore": float(mean_absolute_error(yr_test, reg_pred)),
        "regression_rmse_crore": float(np.sqrt(mean_squared_error(yr_test, reg_pred))),
        "test_rows": int(len(X_test)),
        "warning": "Synthetic demo data; metrics do not represent real-world performance."
    }
    if model_dir:
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out / "classifier.joblib")
        joblib.dump(reg, out / "regressor.joblib")
        joblib.dump(metrics, out / "metrics.joblib")
    return clf, reg, metrics

def predict_movie(inputs, df, model_dir=None):
    clf, reg, _ = train_models(df, model_dir)
    row = pd.DataFrame([{col: inputs[col] for col in FEATURE_COLUMNS}])
    category = str(clf.predict(row)[0])
    probabilities = clf.predict_proba(row)[0]
    classes = list(clf.classes_)
    probability = float(probabilities[classes.index(category)])
    revenue = max(0.0, float(reg.predict(row)[0]))
    return {"category": category, "probability": probability, "revenue_crore": revenue}

def evaluate_models(df):
    return train_models(df)[2]

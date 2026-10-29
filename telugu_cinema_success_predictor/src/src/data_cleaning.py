"""Load, clean and encode the UCI Student Performance dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"
TARGET = "G3"


def make_sample(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic data in the same schema (fallback when real dataset is absent)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[0.8, 0.2]),
        "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(15, 22, n),
        "address": rng.choice(["U", "R"], n, p=[0.75, 0.25]),
        "Medu": rng.integers(0, 5, n),
        "Fedu": rng.integers(0, 5, n),
        "studytime": rng.integers(1, 5, n),
        "failures": rng.choice([0, 1, 2, 3], n, p=[0.8, 0.1, 0.06, 0.04]),
        "schoolsup": rng.choice(["yes", "no"], n, p=[0.12, 0.88]),
        "higher": rng.choice(["yes", "no"], n, p=[0.92, 0.08]),
        "internet": rng.choice(["yes", "no"], n, p=[0.83, 0.17]),
        "goout": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n),
        "absences": rng.poisson(5, n),
    })
    base = 8 + 1.2 * df.studytime - 1.5 * df.failures - 0.08 * df.absences + rng.normal(0, 1.5, n)
    df["G1"] = np.clip(base + rng.normal(0, 1, n), 0, 20).round().astype(int)
    df["G2"] = np.clip(base + rng.normal(0.5, 1, n), 0, 20).round().astype(int)
    df["G3"] = np.clip(0.2 * df.G1 + 0.8 * df.G2 + rng.normal(0, 1, n), 0, 20).round().astype(int)
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load real dataset (semicolon separated). Falls back to synthetic sample."""
    if Path(path).exists():
        return pd.read_csv(path, sep=";")
    print(f"[warn] {path} not found -> using synthetic sample data. "
          "Download student-mat.csv into data/ for real results.")
    return make_sample()


def clean(df: pd.DataFrame, drop_grades: bool = False) -> pd.DataFrame:
    """Remove duplicates, fill missing values, drop invalid targets.

    drop_grades=True removes G1/G2 (early-warning mode: predict from behaviour only).
    """
    df = df.drop_duplicates().copy()
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    df = df[df[TARGET].between(0, 20)]
    if drop_grades:
        df = df.drop(columns=["G1", "G2"], errors="ignore")
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame):
    """One-hot encode categoricals -> (X, y)."""
    y = df[TARGET]
    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    return X, y


def get_xy(drop_grades: bool = False):
    return encode(clean(load_data(), drop_grades=drop_grades))


if __name__ == "__main__":
    X, y = get_xy()
    print("Features:", X.shape, "| Target:", y.shape)

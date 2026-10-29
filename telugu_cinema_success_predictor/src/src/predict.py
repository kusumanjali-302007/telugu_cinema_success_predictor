"""Predict final grade for one student. Usage: python src/predict.py"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def predict(student: dict, tag: str = "with_grades") -> float:
    bundle = joblib.load(ROOT / "models" / f"best_model_{tag}.joblib")
    row = pd.get_dummies(pd.DataFrame([student])).reindex(columns=bundle["columns"], fill_value=0)
    return float(bundle["model"].predict(row)[0])


if __name__ == "__main__":
    demo = {"age": 17, "studytime": 3, "failures": 0, "absences": 2, "G1": 13, "G2": 14}
    print(f"Predicted G3: {predict(demo):.1f} / 20")

"""Train and compare Linear Regression vs Random Forest."""
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from data_cleaning import get_xy

ROOT = Path(__file__).resolve().parent.parent
REPORTS, MODELS = ROOT / "reports", ROOT / "models"


def evaluate(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "MAE": round(mean_absolute_error(y_te, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
        "R2": round(r2_score(y_te, pred), 3),
        "CV_R2": round(cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean(), 3),
    }, pred


def main(drop_grades: bool):
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    X, y = get_xy(drop_grades=drop_grades)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    results, preds = {}, {}
    for name, m in models.items():
        results[name], preds[name] = evaluate(m, X_tr, X_te, y_tr, y_te)

    table = pd.DataFrame(results).T
    mode = "without G1/G2" if drop_grades else "with G1/G2"
    print(f"Mode: {mode}")
    print(table)
    tag = "no_grades" if drop_grades else "with_grades"
    (REPORTS / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))

    best = table["R2"].idxmax()
    joblib.dump({"model": models[best], "columns": list(X.columns)}, MODELS / f"best_model_{tag}.joblib")
    print(f"Best model: {best} (saved to models/)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(y_te, preds[best], alpha=.6, color="#4C72B0")
    ax[0].plot([0, 20], [0, 20], "r--")
    ax[0].set(xlabel="Actual G3", ylabel="Predicted G3", title=f"{best}: Actual vs Predicted")
    imp = pd.Series(models["RandomForest"].feature_importances_, index=X.columns).nlargest(10)[::-1]
    imp.plot.barh(ax=ax[1], color="#55A868"); ax[1].set_title("Top 10 Feature Importances (RF)")
    plt.tight_layout(); plt.savefig(REPORTS / f"model_results_{tag}.png", dpi=150); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-grades", action="store_true", help="exclude G1/G2 (early-warning mode)")
    main(p.parse_args().no_grades)

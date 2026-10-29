from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinema_model import load_dataset, train_models, predict_movie

def test_dataset_and_training():
    df = load_dataset(ROOT / "data" / "telugu_movies.csv")
    assert len(df) >= 20
    _, _, metrics = train_models(df)
    assert 0 <= metrics["classification_accuracy"] <= 1
    assert metrics["regression_mae_crore"] >= 0

def test_prediction():
    df = load_dataset(ROOT / "data" / "telugu_movies.csv")
    row = df.iloc[0]
    keys = ["budget_crore", "runtime_minutes", "release_month", "genre", "director_profile", "cast_profile", "music_profile", "production_profile"]
    result = predict_movie({k: row[k] for k in keys}, df)
    assert result["category"] in set(df["success_category"])
    assert result["revenue_crore"] >= 0
    assert 0 <= result["probability"] <= 1

from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from cinema_model import load_dataset, predict_movie

st.set_page_config(page_title="Telugu Cinema Success Predictor", page_icon="🎬", layout="wide")
st.title("🎬 Telugu Cinema Success Predictor")
st.caption("Telugu cinema · Machine Learning · Streamlit")
st.warning("Important: The included dataset is synthetic demo data, not verified real box-office history. Predictions demonstrate the workflow only and are not reliable for real movie decisions.")

df = load_dataset(ROOT / "data" / "telugu_movies.csv")
predict_tab, explore_tab, about_tab = st.tabs(["🔮 Predict", "📊 Explore data", "ℹ️ About"])

with predict_tab:
    st.subheader("Enter movie details")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        title = c1.text_input("Movie title", "Example Telugu Film")
        budget = c2.number_input("Estimated budget (₹ crore)", 0.0, 1000.0, 30.0, 1.0)
        runtime = c3.number_input("Runtime (minutes)", 30, 300, 150, 5)
        c4, c5, c6 = st.columns(3)
        genre = c4.selectbox("Primary genre", sorted(df["genre"].unique()))
        month = c5.selectbox("Release month", list(range(1, 13)), index=4)
        director = c6.selectbox("Director track record", ["New/limited history", "Established", "High historical performance"])
        c7, c8, c9 = st.columns(3)
        cast = c7.selectbox("Lead cast popularity", ["Emerging", "Established", "High"])
        music = c8.selectbox("Music profile", ["Standard", "Strong audience reach"])
        production = c9.selectbox("Production house profile", ["Independent/small", "Established"])
        submitted = st.form_submit_button("Predict movie outcome", type="primary")
    if submitted:
        inputs = {
            "budget_crore": budget, "runtime_minutes": runtime, "release_month": month,
            "genre": genre, "director_profile": director, "cast_profile": cast,
            "music_profile": music, "production_profile": production,
        }
        try:
            result = predict_movie(inputs, df, ROOT / "models")
            st.subheader(f"Prediction for: {title or 'Untitled movie'}")
            a, b = st.columns(2)
            a.metric("Estimated success category", result["category"])
            b.metric("Estimated gross revenue", f"₹{result['revenue_crore']:.1f} crore")
            st.caption(f"Demo-model class probability: {result['probability']:.0%}; not calibrated or guaranteed.")
            st.info("This output is produced from synthetic demo data. Replace the CSV with verified, sourced records before interpreting results.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with explore_tab:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Demo records", len(df))
    b.metric("Genres", df["genre"].nunique())
    c.metric("Years covered", f"{int(df.release_year.min())}–{int(df.release_year.max())}")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Average demo gross by genre")
    st.bar_chart(df.groupby("genre")["gross_crore"].mean().sort_values(ascending=False))
    st.subheader("Budget versus gross (demo data)")
    st.scatter_chart(df, x="budget_crore", y="gross_crore", color="genre")

with about_tab:
    st.markdown("""
### What is included?
- CSV loading, basic validation, and duplicate removal.
- Classification model: illustrative success category.
- Regression model: estimated gross revenue.
- Interactive Streamlit form and exploratory charts.
- Tests and Windows setup instructions.

### Limitations
- Bundled rows are generated synthetic demo data, not actual Telugu films.
- TMDB metadata is not a verified box-office dataset.
- Wikipedia collection figures can conflict across sources and require validation.
- Success categories are illustrative labels based on gross/budget ratio.
- No external API calls are made by default; the app runs without an API key.
- For real use, gather licensed/credible historical records and evaluate using a chronological holdout set.
""")

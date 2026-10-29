# Telugu Cinema Success Predictor

A beginner-friendly Data Science project using Python, pandas, scikit-learn, and Streamlit. It demonstrates a classification workflow (illustrative success category) and a regression workflow (estimated gross revenue).

> **Important data notice:** The bundled `data/telugu_movies.csv` contains synthetic demo records, not real Telugu film history. The app is runnable immediately, but its predictions are only a software demonstration. Do not use them for real commercial decisions.

## Features
- Streamlit movie-input form.
- Classification: `Hit`, `Average`, or `Flop` based on illustrative gross-to-budget ratios in the demo data.
- Regression: estimated gross revenue in ₹ crore.
- Exploratory charts for genre, budget, gross, and release years.
- Basic data validation, duplicate removal, model metrics, and tests.
- No API key is required for the included demo.

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for installing Python packages
- Windows PowerShell or Command Prompt

## Run on Windows
1. Extract the ZIP.
2. Open the extracted `telugu_cinema_success_predictor` folder.
3. Open Command Prompt/PowerShell in that folder.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

7. Streamlit normally opens the local app in your browser. If not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Run tests
```powershell
pytest -q
```

## Data schema
Replace `data/telugu_movies.csv` with validated rows using these columns:

| Column | Meaning |
|---|---|
| `movie_title` | Film title (metadata only in this demo) |
| `release_year` | Release year |
| `release_month` | Release month (1–12) |
| `genre` | Primary genre |
| `budget_crore` | Budget in ₹ crore |
| `gross_crore` | Gross revenue in ₹ crore |
| `runtime_minutes` | Runtime in minutes |
| `director_profile` | Categorical historical profile |
| `cast_profile` | Categorical cast profile |
| `music_profile` | Categorical music profile |
| `production_profile` | Categorical production-house profile |
| `success_category` | `Hit`, `Average`, or `Flop` |

All profile columns must use categories consistent with the UI, or update the UI choices accordingly.

## Real data: TMDB + box office
- TMDB can provide movie metadata, cast/crew, genres, release dates, and sometimes budget/revenue fields. Coverage and accuracy vary; it is not a complete verified Telugu box-office source.
- Read and comply with TMDB's current API terms, attribution requirements, and rate limits before integrating it.
- Wikipedia is collaboratively edited and box-office numbers may conflict. Keep a source URL, access date, currency, gross/net definition, and confidence note for each value you collect.
- Do not fabricate missing revenue values. Exclude unknown targets from model training and document missingness.
- Never use post-release variables (reviews, final ratings, actual collections) in a pre-release prediction model; that causes target leakage.
- For a serious model, use a time-based train/test split, compare against simple baselines, report metrics, and test performance on a later release period.

## Suggested next improvements
1. Build a sourced and auditable Telugu-film dataset.
2. Add TMDB API ingestion with environment-variable secrets and source attribution.
3. Store box-office source URLs and data-quality notes.
4. Use time-based evaluation and class-specific precision/recall/F1.
5. Deploy to Streamlit Community Cloud after uploading the project to GitHub.

## Project structure
```text
telugu_cinema_success_predictor/
├── app.py
├── data/
│   └── telugu_movies.csv
├── models/                 # models generated at runtime; not bundled
├── src/
│   └── cinema_model.py
├── tests/
│   └── test_model.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

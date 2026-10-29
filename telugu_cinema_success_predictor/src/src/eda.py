"""Exploratory Data Analysis -> saves charts in reports/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import load_data, clean, TARGET

OUT = Path(__file__).resolve().parent.parent / "reports"
OUT.mkdir(exist_ok=True)


def main():
    df = clean(load_data())
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.histplot(df[TARGET], bins=20, kde=True, color="#4C72B0")
    plt.title("Final Grade (G3) Distribution")
    plt.tight_layout(); plt.savefig(OUT / "grade_distribution.png", dpi=150); plt.close()

    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(corr) <= 15, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(OUT / "correlation_heatmap.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(x="studytime", y=TARGET, data=df, ax=ax[0]); ax[0].set_title("Study time vs G3")
    sns.boxplot(x="failures", y=TARGET, data=df, ax=ax[1]); ax[1].set_title("Past failures vs G3")
    sns.scatterplot(x="absences", y=TARGET, data=df, ax=ax[2], alpha=.5); ax[2].set_title("Absences vs G3")
    plt.tight_layout(); plt.savefig(OUT / "key_factors.png", dpi=150); plt.close()

    top = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(5)
    print("Top correlations with G3:")
    print(top.round(3))
    print(f"Charts saved to {OUT}")


if __name__ == "__main__":
    main()

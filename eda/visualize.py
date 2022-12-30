import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd

from eda.constants import FIGURES_DIR, FIG_SIZE
from preprocessing.utils import get_sentiment_score

sns.set(rc={"figure.figsize": FIG_SIZE})


def barplot_top(comments: dd.DataFrame, column: str, year: int, n: int = 10):
    top_comments = comments[[column]].groupby(by=column).size().nlargest(n=n).compute()

    ax = sns.barplot(x=top_comments.index, y=top_comments.values)
    if n > 15:
        ax.set(xticklabels=[])
    ax.set_title(f"Most active {column}s ({year})")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel(column)
    plt.savefig(
        fname=f"{FIGURES_DIR}/barplot_top_{column}_{year}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.show()


def plot_daily_comments(comments: dd.DataFrame, year: int):
    daily_comments = comments[["date"]].groupby(by="date").size().compute()

    ax = sns.lineplot(x=daily_comments.index, y=daily_comments.values)

    ax.set_title(f"Number of daily comments ({year})")
    ax.set_ylabel("Number of comments")
    ax.set_xlabel("Date")
    plt.savefig(
        fname=f"{FIGURES_DIR}/plot_daily_comments_{year}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )

    plt.show()


def plot_daily_sentiment(comments: dd.DataFrame, year: int):
    comments["sentiment"] = comments["body_cleaned"].apply(
        get_sentiment_score, meta=("body_cleaned", "float")
    )
    comments_daily_sentiment = (
        comments[["date", "sentiment"]].groupby(by="date")["sentiment"].mean().compute()
    )

    ax = sns.lineplot(
        x=comments_daily_sentiment.index, y=comments_daily_sentiment.values
    )

    ax.set_title(f"Daily mean sentiment scores ({year})")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    plt.savefig(
        fname=f"{FIGURES_DIR}/plot_daily_sentiment_{year}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )

    plt.show()

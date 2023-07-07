import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from eda.constants import FIGURES_DIR, PARTIES_COLORS
from affection.utils import get_compound_sentiment_score


def barplot_top(comments: pd.DataFrame, column: str, year: int, n: int = 10):
    top_comments = comments[[column]].groupby(by=column).size().nlargest(n=n)

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


def plot_daily_comments(comments: pd.DataFrame, year: int):
    daily_comments = comments[["date"]].groupby(by="date").count()

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


def plot_daily_sentiment(comments: pd.DataFrame, year: int):
    comments["sentiment"] = comments["body_cleaned"].progress_apply(
        get_compound_sentiment_score, meta=("body_cleaned", "float")
    )
    comments_daily_sentiment = (
        comments[["date", "sentiment"]].groupby(by="date")["sentiment"].mean()
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


def plot_event_comments_distribution(comments, theme, event_key):
    plt.figure(figsize=(15, 6))

    sns.histplot(
        data=comments,
        y="event_name",
        weights="number_comments",
        hue="party",
        multiple="stack",
        palette=PARTIES_COLORS,
    )

    plt.title(f"Number of comments per mass shooting and party")
    plt.xlabel("Number of comments")
    plt.ylabel("Event")

    import os

    if not os.path.exists(f"{FIGURES_DIR}/eda/{theme}/{event_key}"):
        os.makedirs(f"{FIGURES_DIR}/eda/{theme}/{event_key}")

    plt.legend(labels=["Republicans", "Democrats"])
    plt.savefig(
        f"{FIGURES_DIR}/eda/{theme}/{event_key}/number_comments_per_event_and_party.pdf",
        bbox_inches="tight",
    )

    plt.show()


# Plot monthly polarization


def plot_polarization_by_time(data, event_key, event_details, granularity):
    plt.figure(figsize=(30, 6))

    ax = sns.lineplot(
        data=data,
        x="date",
        y="polarization",
        label="Polarization",
        marker="o",
        color="k",
    )
    sns.lineplot(
        data=data,
        x="date",
        y="random_polarization",
        label="Polarization with random user assignment",
        color="orange",
    )

    for relevant_event, relevant_event_date in event_details["relevant_dates"].items():
        date_str = relevant_event_date.strftime("%Y-%m-%d")
        plt.axvline(
            relevant_event_date,
            linestyle="--",
            color="blue",
            label=f"{relevant_event} ({date_str})",
        )

    plt.axvline(
        event_details["date"],
        linestyle="--",
        color="red",
        label=f"{event_details['name']} ({date_str})",
    )

    plt.xlabel("Date")
    plt.ylabel("Leave-out partisanship estimation")
    plt.legend()

    plt.savefig(
        fname=f"{FIGURES_DIR}/{event_key}_leaveout_polarization_by_{granularity}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.show()


def plot_nr_users_by_time(
    data, event_key, event_details, granularity,
):
    plt.figure(figsize=(30, 6))

    if granularity == "month":
        _ = sns.barplot(
            x=data["date"],
            y=data["user_cnt"],
            hue=data["party"],
        )
    else:
        _ = sns.lineplot(
            x=data["date"],
            y=data["user_cnt"],
        )

    color = iter(
        cm.Greens(
            np.linspace(0.4, 1, len(event_details["relevant_dates"]))
        )
    )

    for relevant_event, relevant_event_date in event_details[
        "relevant_dates"
    ].items():
        relevant_event_date_str = relevant_event_date.strftime("%Y-%m-%d")
        plt.axvline(
            relevant_event_date,
            linestyle="--",
            color=next(color),
            label=f"{relevant_event} ({relevant_event_date_str})",
        )

    event_date_str = event_details["date"].strftime("%Y-%m-%d")
    plt.axvline(
        event_details["date"],
        linestyle="--",
        color="red",
        label=f"{event_details['name']} ({event_date_str})",
    )

    plt.xlabel("Date")
    plt.ylabel("Number of users")
    plt.legend()

    plt.savefig(
        fname=f"{FIGURES_DIR}/{event_key}_nr_users_by_{granularity}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
    )
    plt.show()


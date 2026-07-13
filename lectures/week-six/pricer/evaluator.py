"""
evaluator.py — a shared test harness for every price model in the project.

Whatever the model is — a coin-flip baseline, a linear regression, a Random Forest,
or later an LLM — we evaluate it the same way: run it over a fixed slice of the test set,
measure the absolute dollar error per item, and report the headline numbers plus two charts.

Usage is deliberately tiny:

    from pricer.evaluator import evaluate
    evaluate(my_pricer, test)   # my_pricer(item) -> predicted price

`evaluate` just wraps `Tester`, which:
  - runs the predictor over `size` items (threaded, since LLM calls are I/O-bound),
  - colours each guess green / orange / red by how close it was,
  - prints a running stream of errors, then shows a cumulative-error trend chart and a
    truth-vs-guess scatter, titled with average error, MSE and r².

`post_process` lets a predictor return a raw string like "Price is $42.50" — we strip
the noise and pull out the number — so the same harness works for ML models and LLMs alike.

Adapted from ed-donner/llm_engineering (week6/pricer/evaluator.py), with docstrings added.
"""

import re
import math
from itertools import accumulate
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm

# ANSI colours for the live error stream printed to the notebook.
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

WORKERS = 5          # threads — predictions are I/O-bound (network) for LLM models
DEFAULT_SIZE = 200   # how many test items to evaluate


class Tester:
    """Runs one predictor over the test set and reports error metrics + charts."""

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []
        self.workers = workers

    @staticmethod
    def make_title(predictor) -> str:
        """Turn a function name like `xg_boost` into a tidy chart title ('Xg Boost')."""
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value):
        """Coerce a prediction to a float — pulling the first number out of a string if needed."""
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0
        else:
            return value

    def color_for(self, error, truth):
        """Green if close (abs or relative), orange if so-so, red if way off."""
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        """Predict one item and return (title, guess, truth, error, color)."""
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = datapoint.price
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        return title, guess, truth, error, color

    def chart(self, title):
        """Scatter of predicted vs actual price, coloured by accuracy, with a y=x reference line."""
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        # Pre-format hover text
        df["hover"] = [
            f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            width=1000,
            height=800,
        )

        # Assign customdata per trace (one color/category = one trace)
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Reference line y=x — perfect predictions land exactly on it.
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self):
        """Cumulative average absolute error as we evaluate more items, with a 95% CI band."""
        n = len(self.errors)

        # Running mean and std (pure Python)
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # 95% confidence interval for mean
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        # Plot
        fig = go.Figure()

        # Shaded confidence interval band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Main line with hover text showing CI
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Avg Error",
                customdata=list(
                    zip(
                        ci,
                    )
                ),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        # Title with final stats
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Error: ${final_mean:,.2f} ± ${final_ci:,.2f}"

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    def report(self):
        """Print/plot the headline metrics: average error, MSE, r², plus the two charts."""
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        title = f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} <b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}%"
        self.error_trend_chart()
        self.chart(title)

    def run(self):
        """Evaluate `size` items in parallel, streaming coloured errors, then report."""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, guess, truth, error, color in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    """Convenience wrapper: build a Tester for `function` over `data` and run it."""
    Tester(function, data, size=size, workers=workers).run()

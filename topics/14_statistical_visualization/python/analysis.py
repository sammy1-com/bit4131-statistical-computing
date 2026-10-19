"""Topic 14: Statistical data visualization (matplotlib, plotly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load

R = Report(__file__)
df = load("students.csv")
sales = load("monthly_sales.csv")
sales["date"] = pd.to_datetime(sales["date"])
colors = {"A": "#d7301f", "B": "#2c7fb8", "C": "#31a354"}

R.h("WORKED EXAMPLE: choosing the right chart")
R.p("One numeric variable        -> histogram / density / boxplot")
R.p("Numeric by category         -> boxplot / violin / bar of means with error bars")
R.p("Two numeric variables       -> scatter plot (+ regression line)")
R.p("Numeric over time           -> line graph")
R.p("Counts of a category        -> bar chart (never a pie chart for many categories)")
R.p("Many numeric variables      -> correlation heatmap / scatter matrix")

R.h("DATASET: 6-panel statistical dashboard")
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
ax[0, 0].hist(df.exam_score, bins=20, color="#9ecae1", edgecolor="white")
ax[0, 0].set(title="Histogram: exam score", xlabel="Score", ylabel="Students")

data = [df[df.method == m].exam_score for m in "ABC"]
bp = ax[0, 1].boxplot(data, tick_labels=list("ABC"), patch_artist=True)
for patch, m in zip(bp["boxes"], "ABC"):
    patch.set_facecolor(colors[m]); patch.set_alpha(0.6)
ax[0, 1].set(title="Boxplot: score by method", xlabel="Method", ylabel="Score")

ax[0, 2].violinplot(data, showmedians=True); ax[0, 2].set_xticks([1, 2, 3]); ax[0, 2].set_xticklabels(list("ABC"))
ax[0, 2].set(title="Violin: score distribution shape", xlabel="Method")

for m in "ABC":
    g = df[df.method == m]
    ax[1, 0].scatter(g.study_hours, g.exam_score, s=14, alpha=0.6, color=colors[m], label=m)
b, a = np.polyfit(df.study_hours, df.exam_score, 1)
xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 20)
ax[1, 0].plot(xs, a + b * xs, "k"); ax[1, 0].legend(title="Method")
ax[1, 0].set(title="Scatter: study hours vs score", xlabel="Study hours", ylabel="Score")

pr = pd.crosstab(df.program, df.passed, normalize="index")[1] * 100
ax[1, 1].bar(pr.index, pr.values, color="#fdae6b"); ax[1, 1].tick_params(axis="x", rotation=15)
ax[1, 1].set(title="Bar: pass rate by programme (%)", ylabel="% passed")

ax[1, 2].plot(sales.date, sales.sales, color="#2c7fb8")
ax[1, 2].set(title="Line: monthly sales", xlabel="Year", ylabel="Sales")
R.save_fig(fig, "py_dashboard.png")
R.p("Saved py_dashboard.png (histogram, boxplot, violin, scatter, bar, line)")

# Scatter matrix
fig = pd.plotting.scatter_matrix(df[["study_hours", "attendance_pct", "prior_gpa", "exam_score"]], figsize=(9, 9), alpha=0.4, diagonal="hist")
plt.suptitle("Scatter matrix")
R.save_fig(plt.gcf(), "py_scatter_matrix.png")

# Bad vs good chart illustration: truncated axis
means = df.groupby("method").exam_score.mean()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(means.index, means.values, color="#bcbddc"); ax[0].set_ylim(55, 67); ax[0].set_title("Misleading: truncated axis")
ax[1].bar(means.index, means.values, color="#9ecae1"); ax[1].set_ylim(0, 80); ax[1].set_title("Honest: axis starts at 0")
R.save_fig(fig, "py_misleading_vs_honest.png")

# Interactive plotly chart (HTML)
import plotly.express as px
fig_p = px.scatter(df, x="study_hours", y="exam_score", color="method", trendline="ols",
                   hover_data=["student_id", "attendance_pct"], title="Interactive: study hours vs exam score")
fig_p.write_html(R.out / "py_interactive_scatter.html")
R.p("Saved py_interactive_scatter.html (open in a browser)")

R.h("INTERPRETATION")
R.p("- Histogram shows an approximately bell-shaped score distribution centred near 61.")
R.p("- Boxplot/violin reveal method C shifts the whole distribution upward while spreads stay similar.")
R.p("- Scatter shows a clear positive linear relationship between study hours and score.")
R.p("- The truncated-axis bar chart exaggerates differences between methods: always start bar axes at zero.")
R.finish()

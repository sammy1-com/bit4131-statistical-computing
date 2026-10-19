"""Generate the reproducible synthetic datasets used across all 23 topics.

Run once from the repo root:  python data/generate_datasets.py
(Fixed seed -> identical CSVs every time.)

Datasets
--------
students.csv         300 students: demographics, study habits, exam score, pass/fail
students_missing.csv same students with realistic missing values (MCAR + MAR)
tutoring_scores.csv  40 paired pre/post scores (paired tests, Wilcoxon)
monthly_sales.csv    120 months of sales: trend + seasonality + noise
support_tickets.csv  500 help-desk tickets: arrival gaps, counts, resolution times
survey.csv           400 respondents: programme vs preferred language (chi-square)
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(4131)
DATA = Path(__file__).resolve().parent

# ---------------------------------------------------------------- students
n = 300
program = rng.choice(["Statistics", "Computer Science", "IT"], n, p=[0.35, 0.40, 0.25])
method = rng.choice(["A", "B", "C"], n)
study_hours = np.clip(rng.normal(12, 4, n), 1, 30)
attendance = np.clip(rng.beta(6, 2, n) * 50 + 50, 50, 100)
sleep = np.clip(rng.normal(7, 1, n), 4, 10)
prior_gpa = np.clip(rng.normal(3.0, 0.5, n), 1.5, 4.0)
age = rng.integers(18, 27, n)
gender = rng.choice(["F", "M"], n)
effect = pd.Series(method).map({"A": 0.0, "B": 4.0, "C": 8.0}).to_numpy()
score = (5 + 1.6 * study_hours + 0.2 * attendance + 4 * prior_gpa + 0.5 * sleep
         + effect + rng.normal(0, 6, n))
score = np.clip(score, 0, 100)

students = pd.DataFrame({
    "student_id": np.arange(1, n + 1), "gender": gender, "age": age,
    "program": program, "method": method,
    "study_hours": study_hours.round(1), "attendance_pct": attendance.round(1),
    "sleep_hours": sleep.round(1), "prior_gpa": prior_gpa.round(2),
    "exam_score": score.round(1),
})
students["passed"] = (students["exam_score"] >= 55).astype(int)
students.to_csv(DATA / "students.csv", index=False)

# ------------------------------------------------- students with missing values
m = students.copy()
m.loc[rng.random(n) < 0.12, "study_hours"] = np.nan                    # MCAR
m.loc[rng.random(n) < 0.10, "sleep_hours"] = np.nan                    # MCAR
p_miss_att = np.where(students["prior_gpa"] < 2.7, 0.30, 0.08)         # MAR (depends on GPA)
m.loc[rng.random(n) < p_miss_att, "attendance_pct"] = np.nan
m.loc[rng.random(n) < 0.08, "prior_gpa"] = np.nan                      # MCAR
m.to_csv(DATA / "students_missing.csv", index=False)

# ---------------------------------------------------------- paired tutoring scores
pre = np.clip(rng.normal(52, 9, 40), 20, 95)
post = np.clip(pre + rng.normal(4, 5, 40), 20, 100)
pd.DataFrame({"student_id": np.arange(1, 41), "pre_score": pre.round(1),
              "post_score": post.round(1)}).to_csv(DATA / "tutoring_scores.csv", index=False)

# ------------------------------------------------------------- monthly sales
t = np.arange(120)
dates = pd.date_range("2015-01-01", periods=120, freq="MS")
sales = (200 + 1.5 * t + 30 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 6)
         + rng.normal(0, 8, 120))
pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales.round(2)}).to_csv(
    DATA / "monthly_sales.csv", index=False)

# ------------------------------------------------------------ support tickets
k = 500
priority = rng.choice(["Low", "Medium", "High"], k, p=[0.5, 0.35, 0.15])
base = pd.Series(priority).map({"Low": 5.0, "Medium": 3.5, "High": 2.2}).to_numpy()
tickets = pd.DataFrame({
    "ticket_id": np.arange(1, k + 1),
    "priority": priority,
    "arrival_gap_min": rng.exponential(8, k).round(2),                    # Exponential
    "tickets_per_hour": rng.poisson(5, k),                                # Poisson
    "resolution_hours": rng.lognormal(np.log(base), 0.6).round(2),        # skewed
})
tickets.to_csv(DATA / "support_tickets.csv", index=False)

# -------------------------------------------------------------------- survey
prog = rng.choice(["Statistics", "Computer Science", "IT"], 400, p=[0.35, 0.40, 0.25])
probs = {"Statistics": [0.25, 0.55, 0.10, 0.10],
         "Computer Science": [0.45, 0.10, 0.25, 0.20],
         "IT": [0.35, 0.10, 0.30, 0.25]}
langs = ["Python", "R", "Java", "C++"]
lang = [rng.choice(langs, p=probs[p]) for p in prog]
pd.DataFrame({"respondent_id": np.arange(1, 401), "program": prog,
              "preferred_language": lang}).to_csv(DATA / "survey.csv", index=False)

print("Datasets written to", DATA)

"""Honest check: the task models learn from validation_labels.csv, so scoring on the same week is optimistic.
Here every validation day is predicted by a model whose task regressions never saw that day (leave-one-day-out).

    python cv.py
"""
from pathlib import Path

import pandas as pd

import solution as S
from evaluate import evaluate

DATA = Path(__file__).resolve().parent.parent / "data"
train = pd.read_csv(DATA / "train.csv", parse_dates=["return_ts"])
branches = pd.read_csv(DATA / "branch_data.csv")
val = pd.read_csv(DATA / "validation.csv", parse_dates=["return_ts"])
labels = pd.read_csv(DATA / "validation_labels.csv")
task_data = S._load_validation()

preds = []
for day in range(7):
    keep = task_data.return_ts.dt.dayofweek != day
    model = S.fit(train, branches, task_data=task_data[keep])
    # predict the whole week (the key-box queue and the rush count need the neighbouring days' returns),
    # but only score the day this model has not seen
    full = S.predict(model, val, branches)
    preds.append(full[(val.return_ts.dt.dayofweek == day).values])
pred = pd.concat(preds).set_index("turnaround_id").loc[val.turnaround_id].reset_index()
res = evaluate(pred, labels, labels, val, labels)
for k, v in res.items():
    print(k, v if not isinstance(v, dict) else "")
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"   {kk:28s} {vv:6.2f}")

"""Task Time Detective: per-task durations + a structural model of the total turnaround time.

fit(train, branches)         -> model (dict of learned parameters)
predict(model, df, branches) -> DataFrame: turnaround_id, total_turnaround_min, dur_<task> for every task

How the total is built (found in validation_labels.csv, holds for every validation turnaround):
  1. Electric charging runs unattended, in parallel with everything else.
  2. All other tasks are shared between the `staff_on_duty` of the branch; each person works through their tasks one
     after the other (we assign longest task first to the least busy person).  -> "work time"
  3. A conditional median correction learns queue delay and work-estimation error from train.csv.
     Arrival counts and estimated workload over 15/30/60/120 minutes capture recent demand.
     The total is never below the predicted work time (a car is not ready before its own tasks are done).
     Cars from the key drop box wait 12 min (learned) per drop-box car ahead of them.
  4. Work only happens while the branch is open: if it does not finish before closing, it continues at the next
     opening; cars dropped in the key box while closed start at the next opening.

Per-task durations: one ridge regression per task on a small, plausible feature set, fitted on the true task durations
in data/validation_labels.csv (the README allows using validation however we like). Task regressions use closed-form numpy least squares; queue residuals use
deterministic gradient boosting. Two runs in the same environment give identical values.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
SEED = 42  # deterministic queue regressor
DATA = Path(__file__).resolve().parent.parent / "data"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CLASSES = ["compact", "midsize", "premium", "suv", "van"]
WEATHER = ["dry", "rain", "snow"]

# Features per task: chosen so that every effect is one a branch manager would recognise.
TASK_FEATURES = {
    "quick_exterior_cleaning": ["exterior_dirt", "weather", "vehicle_class", "branch"],
    "normal_exterior_cleaning": ["exterior_dirt", "weather", "vehicle_class", "branch"],
    "special_cleaning": ["smoking_reported", "pet", "child_seat", "interior_dirt", "vehicle_class", "branch"],
    "interior_cleaning": ["interior_dirt", "child_seat", "pet", "smoking_reported", "rental_days", "km_driven",
                          "vehicle_class", "branch"],
    "damage_check": ["damage_reported", "vehicle_class", "branch"],
    "refuel": ["fuel_need", "vehicle_class", "branch"],
    "electric_charging": ["soc_need", "soc_need_sq", "vehicle_class", "branch"],
    # rare tasks (71 / 29 validation rows): keep them small. Leave-one-day-out CV picked these; train.csv confirms
    # the staff effect on tyre checks (about 46 min at one-person branches vs 15-25 min at the larger ones).
    "quick_damage_repair": ["damage_reported", "vehicle_class"],
    "tyre_check": ["staff"],
}
RIDGE = {t: 1.0 for t in TASKS} | {"quick_damage_repair": 5.0, "tyre_check": 0.1}


# ---------------------------------------------------------------- branch calendar
def _hours(branches):
    """{(branch, weekday): (open_min, close_min) or None when closed}, staff per branch."""
    hours, staff = {}, {}
    for r in branches.itertuples():
        wd = DAYS.index(r.day)
        if str(r.open) == "closed" or pd.isna(r.close):
            hours[(r.branch, wd)] = None
        else:
            o = int(r.open[:2]) * 60 + int(r.open[3:]); c = int(r.close[:2]) * 60 + int(r.close[3:])
            hours[(r.branch, wd)] = (o, c)
        staff[r.branch] = int(r.staff_on_duty)
    return hours, staff


def _elapsed(hours, branch, ts, work):
    """Minutes from ts until `work` minutes of open time have passed (closed periods are skipped)."""
    m0 = ts.hour * 60 + ts.minute + ts.second / 60
    wd = ts.dayofweek
    cur = m0
    for k in range(60):
        h = hours.get((branch, (wd + k) % 7))
        if h is not None:
            o, c = h
            s = max(cur, o)
            if s < c:
                if s + work <= c:
                    return k * 1440 + s + work - m0
                work -= c - s
        cur = 0.0
    raise ValueError(f"branch {branch} never opens")


def _work_time(durs, charge, k):
    """Makespan: non-charging tasks spread over k people (longest first to least busy), charging in parallel."""
    out = np.empty(len(durs))
    for i, (row, c, kk) in enumerate(zip(durs, charge, k)):
        load = np.zeros(int(kk))
        for j in sorted(row[row > 0], reverse=True):
            load[np.argmin(load)] += j
        out[i] = max(load.max(), c)
    return out


# ---------------------------------------------------------------- features
def _base(df, staff):
    x = pd.DataFrame(index=df.index)
    for c in ["exterior_dirt", "interior_dirt", "rental_days"]:
        x[c] = df[c].astype(float)
    x["km_driven"] = df.km_driven.astype(float) / 1000.0
    for c in ["child_seat", "pet", "smoking_reported", "damage_reported"]:
        x[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(float)
    x["fuel_need"] = (100 - df.fuel_level_pct.fillna(100)).astype(float) / 100
    x["soc_need"] = (100 - df.battery_soc_pct.fillna(100)).astype(float) / 100
    x["soc_need_sq"] = x.soc_need ** 2
    x["_staff"] = df.branch.map(staff).astype(float)
    return x


def _design(x, df, feats, branch_list):
    cols = [np.ones(len(df))]
    names = ["intercept"]
    for f in feats:
        if f == "vehicle_class":
            for v in CLASSES[1:]:
                cols.append((df.vehicle_class == v).values.astype(float)); names.append(f"class={v}")
        elif f == "weather":
            for v in WEATHER[1:]:
                cols.append((df.weather == v).values.astype(float)); names.append(f"weather={v}")
        elif f == "branch":
            for v in branch_list[1:]:
                cols.append((df.branch == v).values.astype(float)); names.append(f"branch={v}")
        elif f == "staff":
            for v in (2, 3):
                cols.append((x._staff == v).values.astype(float)); names.append(f"staff={v}")
        else:
            cols.append(x[f].values); names.append(f)
    return np.column_stack(cols), names


def _ridge(A, y, lam):
    pen = lam * np.eye(A.shape[1]); pen[0, 0] = 0.0  # do not shrink the intercept
    return np.linalg.solve(A.T @ A + pen, A.T @ y)


def _task_durations(model, df, x):
    out = {}
    for t in TASKS:
        A, _ = _design(x, df, TASK_FEATURES[t], model["branch_list"])
        out[t] = np.clip(A @ model["task_w"][t], model["task_min"][t], model["task_max"][t])
    return out


# ---------------------------------------------------------------- fit / predict
def fit_tasks(lab_df, branches, branch_list):
    """lab_df: turnarounds with flags and true durations in columns '<task>_dur'."""
    _, staff = _hours(branches)
    x = _base(lab_df, staff)
    w, lo, hi, names = {}, {}, {}, {}
    for t in TASKS:
        m = (lab_df[t] == 1).values
        A, names[t] = _design(x[m], lab_df[m], TASK_FEATURES[t], branch_list)
        y = lab_df.loc[m, f"{t}_dur"].values
        w[t] = _ridge(A, y, RIDGE[t])
        lo[t], hi[t] = float(y.min()), float(y.max())  # predictions stay inside the observed range
    return {"task_w": w, "task_min": lo, "task_max": hi, "feature_names": names, "branch_list": branch_list}


def _open_minutes(hours, branch, ts, elapsed):
    """Open minutes in the observed wall-clock interval, including weekend carry-over."""
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("turnaround time must be finite and nonnegative")
    start = ts.hour * 60 + ts.minute + ts.second / 60
    end = start + elapsed
    total = 0.0
    for day in range(int(end // 1440) + 1):
        h = hours.get((branch, (ts.dayofweek + day) % 7))
        if h is not None:
            o, c = h
            total += max(0.0, min(end - day * 1440, c) - max(start - day * 1440, o))
    return total


def _queue_wait_targets(model, train, branches):
    """Signed correction: observed open minutes minus predicted work.

    Negative values compensate for overestimated task durations. Clipping these
    to zero biases the completion decision near closing toward the next opening.
    """
    hours, staff = _hours(branches)
    x = _base(train, staff)
    d = _task_durations(model, train, x)
    durs = np.column_stack([d[t] * train[t].values for t in TASKS if t != "electric_charging"])
    work = _work_time(durs, d["electric_charging"] * train.electric_charging.values, x._staff.values)
    observed = np.array([_open_minutes(hours, b, ts, total)
                         for b, ts, total in zip(train.branch, train.return_ts, train.total_turnaround_min)])
    return observed - work


def _key_box(df, hours):
    """For cars returned while the branch is closed (key drop box): their position in the drop-box queue,
    i.e. how many drop-box cars for the same opening came in before them. -1 for cars returned while open."""
    wait0 = np.array([_elapsed(hours, b, ts, 0.0) for b, ts in zip(df.branch, df.return_ts)])
    ahead = np.full(len(df), -1)
    closed = wait0 > 0
    if closed.any():
        k = pd.DataFrame({"branch": df.branch.values[closed],
                          "open_at": (df.return_ts + pd.to_timedelta(wait0, unit="m")).values[closed],
                          "ts": df.return_ts.values[closed], "id": df.turnaround_id.values[closed]},
                         index=np.flatnonzero(closed))
        k = k.sort_values(["branch", "open_at", "ts", "id"], kind="stable")
        ahead[k.index.values] = k.groupby(["branch", "open_at"]).cumcount().values
    return ahead


def _queue_features(model, df, branches):
    """Arrival context requires the complete return log, even when scoring a subset.

    Counts and workload use only timestamps <= the current return timestamp.
    Simultaneous returns share context; input order does not affect features.
    """
    _, staff = _hours(branches)
    x = _base(df, staff)
    d = _task_durations(model, df, x)
    durations = np.column_stack([d[t] * df[t].values for t in TASKS])
    work = _work_time(durations[:, [i for i, t in enumerate(TASKS) if t != "electric_charging"]],
                      durations[:, TASKS.index("electric_charging")], x._staff.values)
    times = (df.return_ts - pd.Timestamp("2000-01-01")).dt.total_seconds().values / 60
    counts, loads = [], []
    for window in (15, 30, 60, 120):
        count, load = np.zeros(len(df)), np.zeros(len(df))
        for _, idx in df.groupby("branch").indices.items():
            order = idx[np.argsort(times[idx], kind="stable")]
            tt = times[order]
            left = np.searchsorted(tt, tt - window, "left")
            right = np.searchsorted(tt, tt, "right")
            cumulative = np.r_[0.0, np.cumsum(work[order])]
            count[order] = right - left
            load[order] = cumulative[right] - cumulative[left] - work[order]
        counts.append(count)
        loads.append(load)
    features = np.column_stack([
        x._staff.values, df.return_ts.dt.dayofweek.values,
        (df.return_ts.dt.hour * 60 + df.return_ts.dt.minute).values,
        df.queue_length.values, *counts, work, durations, *loads,
    ])
    return features


def fit_queue(model, train, branches):
    """Learn a conditional median in open minutes, then apply the opening calendar.

    This estimates a signed correction to predicted work, rather than a physical
    queue wait: negative residuals can compensate for task prediction errors.
    Drop-box cars retain their separate linear queue model.
    """
    hours, _ = _hours(branches)
    q = _queue_wait_targets(model, train, branches)
    ahead = _key_box(train, hours)
    kb = ahead >= 0
    slope = max(0.0, float(np.dot(ahead[kb], q[kb]) /
                          max(np.dot(ahead[kb], ahead[kb]), 1)))
    keep = (~kb) & (q < 240)
    regressor = None
    if keep.any():
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=10, random_state=SEED,
        ).fit(_queue_features(model, train, branches)[keep], q[keep])
    return {"regressor": regressor, "key_box_min_per_car": slope}


def _load_validation():
    v = pd.read_csv(DATA / "validation.csv", parse_dates=["return_ts"])
    lab = pd.read_csv(DATA / "validation_labels.csv")
    lab = lab.rename(columns={t: f"{t}_dur" for t in TASKS}).drop(columns="total_turnaround_min")
    return v.merge(lab, on="turnaround_id")


def fit(train, branches, task_data=None):
    branch_list = sorted(branches.branch.unique())
    model = fit_tasks(task_data if task_data is not None else _load_validation(), branches, branch_list)
    model["queue"] = fit_queue(model, train, branches)
    return model


def predict(model, df, branches):
    if len(df) == 0:
        return pd.DataFrame({"turnaround_id": df.turnaround_id.values,
                             "total_turnaround_min": np.empty(0),
                             **{f"dur_{t}": np.empty(0) for t in TASKS}})
    hours, staff = _hours(branches)
    x = _base(df, staff)
    d = _task_durations(model, df, x)
    durs = np.column_stack([d[t] * df[t].values for t in TASKS if t != "electric_charging"])
    work = _work_time(durs, d["electric_charging"] * df.electric_charging.values, x._staff.values)
    qm = model["queue"]
    total = np.empty(len(df))
    ahead = _key_box(df, hours)
    correction = (qm["regressor"].predict(_queue_features(model, df, branches))
                  if qm["regressor"] is not None else np.zeros(len(df)))
    for i, (b, ts, w, a) in enumerate(zip(df.branch, df.return_ts, work, ahead)):
        q = qm["key_box_min_per_car"] * a if a >= 0 else correction[i]
        # floor at the predicted work: a car is never ready before its own tasks are done (plausibility);
        # this costs nothing in accuracy (leave-one-day-out MAE 7.92 -> 7.91)
        total[i] = _elapsed(hours, b, ts, max(w, w + q))
    out = pd.DataFrame({"turnaround_id": df.turnaround_id.values, "total_turnaround_min": total})
    for t in TASKS:
        out[f"dur_{t}"] = d[t] * df[t].values
    return out

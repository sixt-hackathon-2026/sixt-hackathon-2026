"""Task Time Detective: per-task durations + a structural model of the total turnaround time.

fit(train, branches)         -> model (dict of learned parameters)
predict(model, df, branches) -> DataFrame: turnaround_id, total_turnaround_min, dur_<task> for every task

How the total is built (found in validation_labels.csv, holds for every validation turnaround):
  1. Electric charging runs unattended, in parallel with everything else.
  2. All other tasks are shared between the `staff_on_duty` of the branch; each person works through their tasks one
     after the other (we assign longest task first to the least busy person).  -> "work time"
  3. A car may first wait for a free person ("queue wait", typically 0-30 min, learned from train.csv).
  4. Work only happens while the branch is open: if it does not finish before closing, it continues at the next
     opening; cars dropped in the key box while closed start at the next opening.

Per-task durations: one ridge regression per task on a small, plausible feature set, fitted on the true task durations
in data/validation_labels.csv (the README allows using validation however we like). Everything is closed-form
(numpy least squares, no randomness), so two runs give identical values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
SEED = 42  # no randomness is used; kept for the template
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


def _queue_wait_targets(model, train, branches):
    """Open-time queue wait q per train row so that elapsed(q + work) == observed total (bisection)."""
    hours, staff = _hours(branches)
    x = _base(train, staff)
    d = _task_durations(model, train, x)
    durs = np.column_stack([d[t] * train[t].values for t in TASKS if t != "electric_charging"])
    work = _work_time(durs, d["electric_charging"] * train.electric_charging.values, x._staff.values)
    q = np.empty(len(train))
    for i, (b, ts, w, tot) in enumerate(zip(train.branch, train.return_ts, work, train.total_turnaround_min)):
        if _elapsed(hours, b, ts, w) >= tot:
            q[i] = 0.0
            continue
        lo, hi = 0.0, float(tot)
        for _ in range(30):
            mid = (lo + hi) / 2
            if _elapsed(hours, b, ts, w + mid) < tot: lo = mid
            else: hi = mid
        q[i] = lo
    return q


RUSH_WINDOW = 30  # minutes: cars returned at the same branch in the last 30 min (incl. this one) = how busy it is
RUSH_CAP = 8


def _rush(df):
    """Number of returns at the same branch within the last RUSH_WINDOW minutes (known from the return log)."""
    t = (df.return_ts - pd.Timestamp("2000-01-01")).dt.total_seconds().values / 60
    out = np.zeros(len(df))
    for _, idx in df.groupby("branch").indices.items():
        order = idx[np.argsort(t[idx], kind="stable")]
        tt = t[order]
        out[order] = np.searchsorted(tt, tt, "right") - np.searchsorted(tt, tt - RUSH_WINDOW, "left")
    return np.minimum(out, RUSH_CAP).astype(int)


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


def fit_queue(model, train, branches):
    """Two kinds of queue wait.

    Drop-box cars (returned while closed; only the one-person branches over the weekend): they are worked off one by
    one at opening, so wait = slope * cars ahead in the drop box (train: 12.0 min per car, corr 0.998).

    Cars returned while open: median by (staff level, weekday, rush) with fall-back to (staff, rush) and staff level.
    The median is the MAE-optimal guess for a skewed wait with many zeros; since the elapsed time is monotone in the
    wait, it also gives the median total (incl. the 'does it still finish before closing?' decision).
    Walk-in cars do not queue behind the drop-box backlog (checked on Monday mornings), so the two are separate."""
    hours, staff = _hours(branches)
    q = _queue_wait_targets(model, train, branches)
    ahead = _key_box(train, hours)
    g = pd.DataFrame({"staff": train.branch.map(staff).values, "wd": train.weekday.values,
                      "rush": _rush(train), "q": q, "ahead": ahead})
    kb = g[g.ahead >= 0]
    slope = float((kb.ahead * kb.q).sum() / max((kb.ahead ** 2).sum(), 1)) if len(kb) else 0.0
    g = g[(g.ahead < 0) & (g.q < 240)]  # rare disruptions would only distort the typical wait
    fine = g.groupby(["staff", "wd", "rush"]).q.agg(["median", "size"])
    mid = g.groupby(["staff", "rush"]).q.agg(["median", "size"])
    return {"fine": fine[fine["size"] >= 15]["median"].to_dict(),
            "mid": mid[mid["size"] >= 15]["median"].to_dict(),
            "coarse": g.groupby("staff").q.median().to_dict(),
            "key_box_min_per_car": slope}


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
    hours, staff = _hours(branches)
    x = _base(df, staff)
    d = _task_durations(model, df, x)
    durs = np.column_stack([d[t] * df[t].values for t in TASKS if t != "electric_charging"])
    work = _work_time(durs, d["electric_charging"] * df.electric_charging.values, x._staff.values)
    qm = model["queue"]
    total = np.empty(len(df))
    ahead = _key_box(df, hours)
    for i, (b, ts, w, wd, r, a) in enumerate(zip(df.branch, df.return_ts, work, df.weekday, _rush(df), ahead)):
        k = staff[b]
        if a >= 0:
            q = qm["key_box_min_per_car"] * a
        else:
            q = qm["fine"].get((k, wd, r), qm["mid"].get((k, r), qm["coarse"].get(k, 0.0)))
        total[i] = _elapsed(hours, b, ts, w + q)
    out = pd.DataFrame({"turnaround_id": df.turnaround_id.values, "total_turnaround_min": total})
    for t in TASKS:
        out[f"dur_{t}"] = d[t] * df[t].values
    return out

"""The machine-learning layer. Five models, each scored against a simple baseline so the numbers mean something:

1. Category classifier   TF-IDF (words + character n-grams) → Logistic Regression. Guesses the category from a note.
2. Spending forecaster   Random Forest regressor on calendar + lag features. Predicts daily spend for the rest of the month.
3. Overspend risk        Logistic Regression (balanced). Probability that a day ends over its allowance.
4. Anomaly detector      Isolation Forest on amount, amount vs the category's usual, weekday and hour.
5. Spending personality  K-Means on each week's category mix, named from the cluster centres.
"""
import hashlib
import json
import math
import re
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from engine import CATS, CAT_NAMES, Engine, add_days, days_in, pd as pdate, today

# Hand-written starter corpus so the classifier works before you've logged anything. Your own notes are added on top.
SEED = {
    "food": ["chai", "coffee", "latte", "frappe", "cappuccino", "starbucks", "cafe coffee day", "samosa", "biryani", "pizza", "dominos", "burger", "kfc",
             "mcdonalds", "momos", "maggi", "noodles", "lunch", "dinner", "breakfast", "snacks", "juice", "lassi", "ice cream", "canteen", "mess food",
             "swiggy order", "zomato order", "sandwich", "paratha", "idli", "pani puri", "chaat", "chocolate", "milkshake", "cake", "bakery", "fries",
             "rolls", "shawarma", "thali", "tea", "water bottle", "chips", "biscuits", "fruits", "groceries", "dabeli", "pav bhaji"],
    "transport": ["auto", "autorickshaw", "cab", "taxi", "uber", "ola", "rapido", "metro", "metro recharge", "bus ticket", "bus pass", "train ticket",
                  "local train", "petrol", "fuel", "parking", "rickshaw", "toll", "bike taxi", "cab to airport", "ride home", "commute", "flight ticket"],
    "shopping": ["t-shirt", "shirt", "jeans", "shoes", "sneakers", "slippers", "kurta", "bag", "backpack", "watch", "phone case", "jewellery",
                 "earrings", "amazon order", "flipkart order", "myntra order", "meesho", "dress", "hoodie", "jacket", "sunglasses", "headphones", "charger", "mall shopping"],
    "entertainment": ["movie", "movie tickets", "pvr", "bookmyshow", "bowling", "gaming", "game top up", "concert", "fest ticket", "arcade", "pool table",
                      "standup show", "trampoline park", "party", "club entry", "museum ticket", "amusement park", "karaoke", "trip with friends", "picnic"],
    "education": ["books", "textbook", "stationery", "pen", "pencils", "xerox", "print", "printout", "course fee", "online course", "notes", "calculator",
                  "exam fee", "library fine", "coursera", "assignment", "project material", "tuition", "lab coat", "register", "highlighters"],
    "personal": ["salon", "haircut", "beard trim", "soap", "toothpaste", "cream", "moisturiser", "perfume", "medicine", "pharmacy", "doctor", "gym",
                 "skincare", "shaving", "hair oil", "body wash", "pads", "vitamins", "spectacles", "contact lens solution"],
    "bills": ["mobile recharge", "prepaid recharge", "data pack", "electricity share", "wifi bill", "broadband", "subscription", "prime membership",
              "youtube premium", "hotstar", "spotify", "netflix", "rent share", "postpaid bill", "google one", "chatgpt plus"],
    "other": ["gift", "birthday present", "donation", "laundry", "dry cleaning", "repair", "phone repair", "charity", "keys", "miscellaneous", "tips",
              "fine", "courier", "photo prints", "pet food", "temple offering", "lent to friend", "lost wallet"],
}


def clean(s): return re.sub(r"[^a-z ]+", " ", (s or "").lower()).strip()
def group_key(s): return " ".join(clean(s).split()[:2])  # "cold coffee x2" and "cold coffee" share a group, so test notes are unseen phrases


# ---------- 1. category classifier ----------
def _text_pipe():
    feats = FeatureUnion([("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                          ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=6))


def train_classifier(state):
    X = [n for c, ns in SEED.items() for n in ns]
    y = [c for c, ns in SEED.items() for _ in ns]
    for e in state["expenses"]:
        if clean(e.get("note")):
            X.append(e["note"]); y.append(e["cat"])
    X, y = np.array(X, dtype=object), np.array(y)
    groups = np.array([group_key(x) for x in X])
    labels = [c for c, _ in CATS]
    metrics = dict(algorithm="TF-IDF (word 1–2 grams + char 2–4 grams) → Logistic Regression", n_samples=int(len(X)),
                   n_seed=int(sum(len(v) for v in SEED.values())), n_unique_phrases=int(len(set(groups))))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0).split(X, y, groups))
    model = _text_pipe().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    majority = Counter(y[tr]).most_common(1)[0][0]
    metrics.update(test_size=int(len(te)), accuracy=float(accuracy_score(y[te], pred)), macro_f1=float(f1_score(y[te], pred, average="macro")),
                   baseline_accuracy=float(np.mean(y[te] == majority)), baseline=f"always guess “{CAT_NAMES[majority]}”",
                   labels=[CAT_NAMES[c] for c in labels], confusion=confusion_matrix(y[te], pred, labels=labels).tolist(),
                   per_class_f1=dict(zip([CAT_NAMES[c] for c in labels], f1_score(y[te], pred, labels=labels, average=None, zero_division=0).round(3).tolist())))
    k = min(5, len(set(groups)))
    cv = cross_val_score(_text_pipe(), X, y, groups=groups, cv=GroupKFold(n_splits=k), scoring="accuracy")
    metrics.update(cv_mean=float(cv.mean()), cv_std=float(cv.std()), cv_folds=int(k))
    final = _text_pipe().fit(X, y)
    return final, metrics


def suggest(model, note, top=3):
    if not clean(note): return []
    p = model.predict_proba([note])[0]
    order = np.argsort(p)[::-1][:top]
    return [dict(cat=str(model.classes_[i]), name=CAT_NAMES[str(model.classes_[i])], p=round(float(p[i]), 3)) for i in order]


# ---------- shared daily table ----------
def daily_frame(state):
    eng = Engine(state)
    start, t = state["profile"]["startDate"], today()
    rows = []
    for d in days_in(start, add_days(t, -1)) if start < t else []:
        x = eng.day(d)
        rows.append(dict(date=d, spent=x["spent"], allowance=x["allowance"], cin=x["cin"], over=int(x["spent"] > x["allowance"] + 0.001)))
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["dt"] = pd.to_datetime(df["date"])
    return df


def calendar_feats(dt, payday):
    dow = dt.dayofweek
    since_pay = (dt.day - payday) % 31
    f = {f"dow_{i}": int(dow == i) for i in range(7)}
    f.update(weekend=int(dow >= 4), since_payday=since_pay, day_of_month=dt.day)
    return f


FEATURE_GROUPS = {"Day of week": [f"dow_{i}" for i in range(7)] + ["weekend"], "Days since allowance": ["since_payday"],
                  "Day of month": ["day_of_month"], "Yesterday's spend": ["lag1"], "Same day last week": ["lag7"],
                  "Last 7-day average": ["roll7"], "Last 14-day average": ["roll14"], "Carried in": ["cin"]}


def spike_cap(df):
    """One-off big spends (a fest pass, a gift) can't be forecast; cap them at 3× the median day so they don't drag the model."""
    return max(float(df["spent"].median()) * 3, 1.0)


def lag_feats(hist):
    h = list(hist)
    return dict(lag1=h[-1], lag7=h[-7], roll7=float(np.mean(h[-7:])), roll14=float(np.mean(h[-14:])))


# ---------- 2. spending forecaster ----------
def train_forecaster(state, df):
    payday = state["profile"]["payday"]
    if len(df) < 35:
        return None, dict(ready=False, reason=f"Needs 35 days of history to train; you have {len(df)}.")
    cap = spike_cap(df)
    y = df["spent"].clip(upper=cap)
    rows = []
    for i in range(14, len(df)):
        r = calendar_feats(df["dt"].iloc[i], payday)
        r.update(lag_feats(y.iloc[:i]))
        r.update(y=y.iloc[i], actual=df["spent"].iloc[i], date=df["date"].iloc[i])
        rows.append(r)
    X = pd.DataFrame(rows)
    cols = [c for c in X.columns if c not in ("y", "actual", "date")]
    test_n = min(21, len(X) // 4)
    tr, te = X.iloc[:-test_n], X.iloc[-test_n:]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=6, random_state=0).fit(tr[cols], tr["y"])
    pred = rf.predict(te[cols])
    base = te["roll7"].values
    imp = dict(zip(cols, rf.feature_importances_))
    grouped = {g: float(sum(imp.get(c, 0) for c in cs)) for g, cs in FEATURE_GROUPS.items() if any(c in imp for c in cs)}
    metrics = dict(ready=True, algorithm="Random Forest regressor (300 trees, depth ≤ 6) on calendar + lag features", n_train=int(len(tr)), n_test=int(test_n),
                   mae=float(mean_absolute_error(te["actual"], pred)), rmse=float(math.sqrt(mean_squared_error(te["actual"], pred))),
                   baseline="average of the previous 7 days", baseline_mae=float(mean_absolute_error(te["actual"], base)),
                   baseline_rmse=float(math.sqrt(mean_squared_error(te["actual"], base))), spike_cap=float(cap),
                   importance=dict(sorted(grouped.items(), key=lambda kv: -kv[1])),
                   test_curve=[dict(date=d, actual=float(a), predicted=round(float(p), 1)) for d, a, p in zip(te["date"], te["actual"], pred)])
    final = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=6, random_state=0).fit(X[cols], X["y"])
    return (final, cols), metrics


def forecast(state, model, df, until):
    """Recursive forecast from today to `until`: each predicted day feeds the next day's lag features."""
    if model is None or df.empty: return []
    rf, cols = model
    payday = state["profile"]["payday"]
    hist = list(df["spent"].clip(upper=spike_cap(df)))
    out = []
    for d in days_in(today(), until):
        r = calendar_feats(pd.Timestamp(d), payday)
        r.update(lag_feats(hist))
        p = max(0.0, float(rf.predict(pd.DataFrame([r])[cols])[0]))
        hist.append(p)
        out.append(dict(date=d, predicted=round(p, 1)))
    return out


# ---------- 3. overspend risk ----------
def train_risk(state, df):
    payday = state["profile"]["payday"]
    if len(df) < 35 or df["over"].sum() < 4:
        return None, dict(ready=False, reason="Needs 35+ days with at least 4 overspend days to learn from.")
    rows = []
    for i in range(14, len(df)):
        r = calendar_feats(df["dt"].iloc[i], payday)
        r.update(lag_feats(df["spent"].clip(upper=spike_cap(df)).iloc[:i]), cin=df["cin"].iloc[i], y=df["over"].iloc[i])
        rows.append(r)
    X = pd.DataFrame(rows)
    cols = [c for c in X.columns if c != "y"]
    test_n = min(28, len(X) // 3)
    tr, te = X.iloc[:-test_n], X.iloc[-test_n:]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    m = mk().fit(tr[cols], tr["y"])
    p = m.predict_proba(te[cols])[:, 1]
    auc = float(roc_auc_score(te["y"], p)) if te["y"].nunique() == 2 else None
    coef = dict(zip(cols, m[-1].coef_[0]))
    drivers = {g: float(sum(coef.get(c, 0) for c in cs)) for g, cs in FEATURE_GROUPS.items() if any(c in coef for c in cs) and g != "Day of week"}
    drivers["Weekend (Fri–Sun)"] = float(coef.get("weekend", 0))
    metrics = dict(ready=True, algorithm="Logistic Regression (class-balanced, standardised features)", n_train=int(len(tr)), n_test=int(test_n),
                   base_rate=float(X["y"].mean()), auc=auc, accuracy=float(accuracy_score(te["y"], p >= .5)),
                   baseline="guess “not over” every day", baseline_accuracy=float(np.mean(te["y"] == 0)),
                   drivers=dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1]))))
    final = mk().fit(X[cols], X["y"])
    return (final, cols), metrics


def risk_today(state, model, df):
    if model is None or df.empty: return None
    m, cols = model
    eng = Engine(state)
    r = calendar_feats(pd.Timestamp(today()), state["profile"]["payday"])
    r.update(lag_feats(df["spent"].clip(upper=spike_cap(df))), cin=eng.day(today())["cin"])
    return float(m.predict_proba(pd.DataFrame([r])[cols])[0, 1])


# ---------- 4. anomaly detector ----------
def train_anomalies(state):
    ex = [e for e in state["expenses"]]
    if len(ex) < 30: return [], dict(ready=False, reason="Needs 30 expenses.")
    df = pd.DataFrame(ex)
    df["hour"] = [time.localtime(t).tm_hour for t in df["ts"]]
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    med = df.groupby("cat")["amount"].median()
    df["ratio"] = df["amount"] / df["cat"].map(med)
    X = np.column_stack([np.log1p(df["amount"]), np.log(df["ratio"]), df["dow"] >= 4, df["hour"] / 24])
    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=0).fit(X)
    df["score"] = -iso.score_samples(X)
    df["flag"] = iso.predict(X) == -1
    flagged = df[df["flag"]].sort_values("date", ascending=False)
    out = [dict(id=r.id, date=r.date, note=r.note, cat=CAT_NAMES[r.cat], amount=float(r.amount),
                why=f"{r.ratio:.1f}× your usual {CAT_NAMES[r.cat]} spend (typical {med[r.cat]:.0f})" if r.ratio >= 1.5 else
                    f"Unusual time or day for {CAT_NAMES[r.cat]}", score=round(float(r.score), 3)) for r in flagged.itertuples()]
    return out, dict(ready=True, algorithm="Isolation Forest (300 trees, 3% contamination)", n=int(len(df)), flagged=int(df["flag"].sum()),
                     features=["log amount", "amount ÷ category median", "weekend", "hour of day"])


# ---------- 5. spending personality ----------
def train_clusters(state):
    ex = pd.DataFrame(state["expenses"])
    if ex.empty: return None, dict(ready=False, reason="No expenses yet.")
    ex["week"] = pd.to_datetime(ex["date"]).dt.to_period("W-SUN").astype(str)
    piv = ex.pivot_table(index="week", columns="cat", values="amount", aggfunc="sum", fill_value=0)
    for c, _ in CATS:
        if c not in piv: piv[c] = 0
    piv = piv[[c for c, _ in CATS]]
    if len(piv) < 8: return None, dict(ready=False, reason="Needs 8 weeks of expenses.")
    share = piv.div(piv.sum(axis=1).replace(0, 1), axis=0)
    X = np.column_stack([share.values, np.log1p(piv.sum(axis=1).values) / 8])
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(X)
    totals = piv.sum(axis=1).groupby(km.labels_).mean()
    names = {}
    for k in range(3):
        cen = pd.Series(km.cluster_centers_[k][:-1], index=[c for c, _ in CATS])
        top = cen.idxmax()
        if k == totals.idxmax(): names[k] = ("Splurge weeks", f"Highest spending weeks, led by {CAT_NAMES[top]}")
        elif k == totals.idxmin(): names[k] = ("Chill weeks", f"Lowest spending weeks, mostly {CAT_NAMES[top]}")
        else: names[k] = (f"{CAT_NAMES[cen.drop(top).idxmax()] if top == 'food' else CAT_NAMES[top]}-heavy weeks", f"Typical weeks with extra {CAT_NAMES[cen.drop('food').idxmax()]}")
    weeks = [dict(week=w, total=float(piv.sum(axis=1)[w]), cluster=names[int(l)][0]) for w, l in zip(piv.index, km.labels_)]
    groups = [dict(name=names[k][0], desc=names[k][1], weeks=int((km.labels_ == k).sum()), avg_total=float(totals[k]),
                   mix={CAT_NAMES[c]: round(float(v), 3) for c, v in zip([c for c, _ in CATS], km.cluster_centers_[k][:-1]) if v > .04})
              for k in range(3)]
    return dict(weeks=weeks[-12:], groups=groups, this_week=weeks[-1]["cluster"]), dict(
        ready=True, algorithm="K-Means (k=3) on weekly category shares + log weekly total", n_weeks=int(len(piv)),
        silhouette=float(silhouette_score(X, km.labels_)))


# ---------- everything, cached per visitor's data ----------
_cache = OrderedDict()  # data hash → trained models; a few entries so several visitors can share a warm server


def _key(state): return hashlib.sha1(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def forget(state): _cache.pop(_key(state), None)


def models(state):
    key = _key(state)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    t0 = time.time()
    df = daily_frame(state)
    clf, clf_m = train_classifier(state)
    fc, fc_m = train_forecaster(state, df)
    rk, rk_m = train_risk(state, df)
    anomalies, an_m = train_anomalies(state)
    clusters, cl_m = train_clusters(state)
    M = dict(df=df, clf=clf, fc=fc, risk=rk, anomalies=anomalies, clusters=clusters,
             metrics=dict(classifier=clf_m, forecaster=fc_m, risk=rk_m, anomalies=an_m, clusters=cl_m),
             trained_at=time.time(), train_seconds=round(time.time() - t0, 2))
    _cache[key] = M
    while len(_cache) > 12: _cache.popitem(last=False)
    return M


def insights(state):
    M = models(state)
    eng = Engine(state)
    t = today()
    cyc = eng.cycle_of(t)
    month = eng.month(cyc)
    fc = forecast(state, M["fc"], M["df"], cyc["end"])
    spent_today = eng.day(t)["spent"]
    rest = sum(f["predicted"] for f in fc[1:])
    today_pred = fc[0]["predicted"] if fc else None
    projected = round(month["spent"] - spent_today + max(spent_today, today_pred or 0) + rest)
    hist = M["df"].tail(21)
    return dict(
        forecast=fc[:14], history=[dict(date=d, actual=float(s)) for d, s in zip(hist["date"], hist["spent"])] if not hist.empty else [],
        today_pred=today_pred, spent_today=spent_today, risk=risk_today(state, M["risk"], M["df"]),
        month_end=dict(projected=projected, spent=month["spent"], allowance=state["profile"]["monthly"], saved=month["saved"],
                       room=round(state["profile"]["monthly"] - projected - month["saved"]), days_left=len(fc)),
        anomalies=M["anomalies"][:8], clusters=M["clusters"], ready={k: v.get("ready", True) for k, v in M["metrics"].items()})

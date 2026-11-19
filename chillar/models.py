"""The five machine-learning models, each evaluated on unseen data against a simple baseline.

1. Category classifier   TF-IDF (word 1–2 grams + char 2–4 grams) → Logistic Regression     supervised · classification
2. Spending forecaster   Random Forest regressor on calendar + lag features                  supervised · regression
3. Overspend risk        Logistic Regression (class-balanced, standardised)                  supervised · classification
4. Anomaly detector      Isolation Forest                                                     unsupervised
5. Week types            K-Means (k = 3) on each week's category mix                          unsupervised
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CATEGORIES

# Hand-written starter corpus so the classifier knows common words before seeing any of the dataset's notes.
SEED = {
    "Food & snacks": ["chai", "coffee", "latte", "frappe", "cappuccino", "starbucks", "cafe coffee day", "samosa", "biryani", "pizza", "dominos", "burger", "kfc",
             "mcdonalds", "momos", "maggi", "noodles", "lunch", "dinner", "breakfast", "snacks", "juice", "lassi", "ice cream", "canteen", "mess food",
             "swiggy order", "zomato order", "sandwich", "paratha", "idli", "pani puri", "chaat", "chocolate", "milkshake", "cake", "bakery", "fries",
             "rolls", "shawarma", "thali", "tea", "water bottle", "chips", "biscuits", "fruits", "groceries", "dabeli", "pav bhaji"],
    "Transport": ["auto", "autorickshaw", "cab", "taxi", "uber", "ola", "rapido", "metro", "metro recharge", "bus ticket", "bus pass", "train ticket",
                  "local train", "petrol", "fuel", "parking", "rickshaw", "toll", "bike taxi", "cab to airport", "ride home", "commute", "flight ticket"],
    "Shopping": ["t-shirt", "shirt", "jeans", "shoes", "sneakers", "slippers", "kurta", "bag", "backpack", "watch", "phone case", "jewellery",
                 "earrings", "amazon order", "flipkart order", "myntra order", "meesho", "dress", "hoodie", "jacket", "sunglasses", "headphones", "charger", "mall shopping"],
    "Fun & outings": ["movie", "movie tickets", "pvr", "bookmyshow", "bowling", "gaming", "game top up", "concert", "fest ticket", "arcade", "pool table",
                      "standup show", "trampoline park", "party", "club entry", "museum ticket", "amusement park", "karaoke", "trip with friends", "picnic"],
    "Study": ["books", "textbook", "stationery", "pen", "pencils", "xerox", "print", "printout", "course fee", "online course", "notes", "calculator",
                  "exam fee", "library fine", "coursera", "assignment", "project material", "tuition", "lab coat", "register", "highlighters"],
    "Personal care": ["salon", "haircut", "beard trim", "soap", "toothpaste", "cream", "moisturiser", "perfume", "medicine", "pharmacy", "doctor", "gym",
                 "skincare", "shaving", "hair oil", "body wash", "pads", "vitamins", "spectacles", "contact lens solution"],
    "Bills & recharges": ["mobile recharge", "prepaid recharge", "data pack", "electricity share", "wifi bill", "broadband", "subscription", "prime membership",
              "youtube premium", "hotstar", "spotify", "netflix", "rent share", "postpaid bill", "google one", "chatgpt plus"],
    "Other": ["gift", "birthday present", "donation", "laundry", "dry cleaning", "repair", "phone repair", "charity", "keys", "miscellaneous", "tips",
              "fine", "courier", "photo prints", "pet food", "temple offering", "lent to friend", "lost wallet"],
}


def clean(s): return re.sub(r"[^a-z ]+", " ", (s or "").lower()).strip()


def group_key(s):
    """'cold coffee x2' and 'cold coffee' share a group, so the test set only contains phrases never seen in training."""
    return " ".join(clean(s).split()[:2])


# ======================= 1. Category classifier =======================
def text_pipeline():
    feats = FeatureUnion([("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                          ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=6))


def train_classifier(df):
    X = [n for ns in SEED.values() for n in ns] + list(df["note"])
    y = [c for c, ns in SEED.items() for _ in ns] + list(df["category"])
    X, y = np.array(X, dtype=object), np.array(y)
    groups = np.array([group_key(x) for x in X])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=0).split(X, y, groups))
    model = text_pipeline().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    majority = Counter(y[tr]).most_common(1)[0][0]
    k = min(5, len(set(groups)))
    cv = cross_val_score(text_pipeline(), X, y, groups=groups, cv=GroupKFold(n_splits=k), scoring="accuracy")
    return dict(
        model=text_pipeline().fit(X, y), n_samples=len(X), n_seed=sum(len(v) for v in SEED.values()), n_phrases=len(set(groups)),
        test_size=len(te), accuracy=accuracy_score(y[te], pred), macro_f1=f1_score(y[te], pred, average="macro"),
        baseline_accuracy=float(np.mean(y[te] == majority)), baseline=f"always guess “{majority}”",
        labels=CATEGORIES, confusion=confusion_matrix(y[te], pred, labels=CATEGORIES),
        per_class_f1=dict(zip(CATEGORIES, f1_score(y[te], pred, labels=CATEGORIES, average=None, zero_division=0))),
        cv_mean=cv.mean(), cv_std=cv.std(), cv_folds=k)


def predict_category(model, note, top=3):
    if not clean(note): return []
    p = model.predict_proba([note])[0]
    return [(str(model.classes_[i]), float(p[i])) for i in np.argsort(p)[::-1][:top]]


# ======================= shared features for the daily models =======================
FEATURE_GROUPS = {"Day of week": [f"dow_{i}" for i in range(7)] + ["weekend"], "Days since allowance": ["since_payday"],
                  "Day of month": ["day_of_month"], "Yesterday's spend": ["lag1"], "Same day last week": ["lag7"],
                  "Last 7-day average": ["roll7"], "Last 14-day average": ["roll14"]}


def spike_cap(spent):
    """One-off big spends can't be forecast from the calendar; cap them at 3× the median day so they don't drag the fit."""
    return max(float(np.median(spent)) * 3, 1.0)


def features(day, history, payday=1):
    """Calendar features for `day` plus lag features computed only from earlier days (no look-ahead)."""
    h = list(history)
    f = {f"dow_{i}": int(day.dayofweek == i) for i in range(7)}
    f.update(weekend=int(day.dayofweek >= 4), since_payday=(day.day - payday) % 31, day_of_month=day.day,
             lag1=h[-1], lag7=h[-7], roll7=float(np.mean(h[-7:])), roll14=float(np.mean(h[-14:])))
    return f


def feature_table(days, payday=1):
    capped = days["spent"].clip(upper=spike_cap(days["spent"]))
    rows = [features(days.index[i], capped.iloc[:i], payday) for i in range(14, len(days))]
    X = pd.DataFrame(rows, index=days.index[14:])
    return X, capped.iloc[14:], days["spent"].iloc[14:], days["over"].iloc[14:]


def grouped(values, cols):
    named = dict(zip(cols, values))
    return {g: float(sum(named.get(c, 0) for c in cs)) for g, cs in FEATURE_GROUPS.items()}


# ======================= 2. Spending forecaster =======================
def forest(): return RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_depth=6, random_state=0)


def train_forecaster(days, payday=1, test_days=21):
    X, y, actual, _ = feature_table(days, payday)
    tr, te = slice(None, -test_days), slice(-test_days, None)
    rf = forest().fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    base = X["roll7"][te].values
    return dict(
        model=forest().fit(X, y), cols=list(X.columns), n_train=len(X) - test_days, n_test=test_days, cap=spike_cap(days["spent"]),
        mae=mean_absolute_error(actual[te], pred), rmse=math.sqrt(mean_squared_error(actual[te], pred)),
        baseline="average of the previous 7 days", baseline_mae=mean_absolute_error(actual[te], base),
        baseline_rmse=math.sqrt(mean_squared_error(actual[te], base)),
        importance=dict(sorted(grouped(rf.feature_importances_, X.columns).items(), key=lambda kv: -kv[1])),
        test=pd.DataFrame(dict(actual=actual[te].values, predicted=pred, baseline=base), index=X.index[te]))


def forecast(fc, days, horizon=14, payday=1):
    """Recursive forecast: each predicted day becomes history for the next day's lag features."""
    hist = list(days["spent"].clip(upper=fc["cap"]))
    out = []
    for i in range(1, horizon + 1):
        d = days.index[-1] + pd.Timedelta(days=i)
        p = max(0.0, float(fc["model"].predict(pd.DataFrame([features(d, hist, payday)])[fc["cols"]])[0]))
        hist.append(p)
        out.append((d, p))
    return pd.DataFrame(out, columns=["date", "predicted"]).set_index("date")


# ======================= 3. Overspend risk =======================
def risk_pipeline(): return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def train_risk(days, payday=1, test_days=28):
    X, _, _, y = feature_table(days, payday)
    tr, te = slice(None, -test_days), slice(-test_days, None)
    m = risk_pipeline().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    both = y[te].nunique() == 2
    fpr, tpr, _ = roc_curve(y[te], p) if both else ([0, 1], [0, 1], None)
    coef = dict(zip(X.columns, m[-1].coef_[0]))
    drivers = {g: v for g, v in grouped(m[-1].coef_[0], X.columns).items() if g != "Day of week"}
    drivers["Weekend (Fri–Sun)"] = float(coef["weekend"])
    return dict(
        model=risk_pipeline().fit(X, y), cols=list(X.columns), n_train=len(X) - test_days, n_test=test_days, base_rate=float(y.mean()),
        auc=roc_auc_score(y[te], p) if both else None, accuracy=accuracy_score(y[te], p >= .5),
        baseline="guess “not over” every day", baseline_accuracy=float(np.mean(y[te] == 0)), fpr=fpr, tpr=tpr,
        drivers=dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1]))))


def risk_ahead(rk, fc, days, horizon=7, payday=1):
    """Probability of going over budget on each of the next days, using the forecaster's predictions as history."""
    hist = list(days["spent"].clip(upper=fc["cap"]))
    out = []
    for i in range(1, horizon + 1):
        d = days.index[-1] + pd.Timedelta(days=i)
        f = pd.DataFrame([features(d, hist, payday)])
        out.append((d, float(rk["model"].predict_proba(f[rk["cols"]])[0, 1])))
        hist.append(max(0.0, float(fc["model"].predict(f[fc["cols"]])[0])))
    return pd.DataFrame(out, columns=["date", "risk"]).set_index("date")


# ======================= 4. Anomaly detector =======================
def detect_anomalies(df, contamination=.03):
    d = df.copy()
    med = d.groupby("category")["amount"].median()
    d["ratio"] = d["amount"] / d["category"].map(med)
    X = np.column_stack([np.log1p(d["amount"]), np.log(d["ratio"]), d["date"].dt.dayofweek >= 4, d["hour"] / 24])
    iso = IsolationForest(n_estimators=120, contamination=contamination, random_state=0).fit(X)
    d["score"] = -iso.score_samples(X)
    d["anomaly"] = iso.predict(X) == -1
    d["reason"] = np.where(d["ratio"] >= 1.5, d["ratio"].map("{:.1f}× the usual".format) + " " + d["category"] + " spend",
                           "unusual time or day for " + d["category"])
    return d


# ======================= 5. Week types =======================
def week_types(df, k=3):
    w = df.assign(week=df["date"].dt.to_period("W-SUN")).pivot_table(index="week", columns="category", values="amount", aggfunc="sum", fill_value=0)
    w = w.reindex(columns=CATEGORIES, fill_value=0)
    total = w.sum(axis=1)
    share = w.div(total.replace(0, 1), axis=0)
    X = np.column_stack([share.values, np.log1p(total.values) / 8])
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
    avg = total.groupby(km.labels_).mean()
    names = {}
    for c in range(k):
        centre = pd.Series(km.cluster_centers_[c][:-1], index=CATEGORIES)
        if c == avg.idxmax(): names[c] = "Splurge weeks"
        elif c == avg.idxmin(): names[c] = "Chill weeks"
        else: names[c] = f"{centre.drop('Food & snacks').idxmax()}-heavy weeks"
    groups = pd.DataFrame([dict(name=names[c], weeks=int((km.labels_ == c).sum()), avg_total=float(avg[c]),
                                **dict(zip(CATEGORIES, km.cluster_centers_[c][:-1]))) for c in range(k)])
    weeks = pd.DataFrame(dict(week=w.index.astype(str), total=total.values, type=[names[l] for l in km.labels_]))
    return dict(groups=groups, weeks=weeks, silhouette=silhouette_score(X, km.labels_), n_weeks=len(w))


def compare_seeds(seeds=(7, 11, 23, 5, 42), days=120, budget=150, payday=1):
    """Re-run the three supervised models on independently generated datasets to see how stable the results are."""
    from .data import daily, generate
    rows = []
    for sd in seeds:
        df = generate(days, sd, payday); dd = daily(df, budget, days)
        c, f, r = train_classifier(df), train_forecaster(dd, payday), train_risk(dd, payday)
        rows.append(dict(seed=sd, expenses=len(df), clf_acc=c["accuracy"], clf_base=c["baseline_accuracy"], clf_f1=c["macro_f1"], clf_cv=c["cv_mean"],
                         clf_cv_std=c["cv_std"], fc_mae=f["mae"], fc_base_mae=f["baseline_mae"], fc_rmse=f["rmse"], fc_base_rmse=f["baseline_rmse"],
                         risk_auc=r["auc"], risk_acc=r["accuracy"], risk_base=r["baseline_accuracy"], risk_rate=r["base_rate"]))
    return pd.DataFrame(rows)


# ======================= everything at once =======================
def run_all(df, days, payday=1):
    fc = train_forecaster(days, payday)
    rk = train_risk(days, payday)
    return dict(classifier=train_classifier(df), forecaster=fc, risk=rk, anomalies=detect_anomalies(df), weeks=week_types(df),
                forecast=forecast(fc, days, payday=payday), risk_ahead=risk_ahead(rk, fc, days, payday=payday))

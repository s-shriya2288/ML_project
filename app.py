"""Chillar ML: five machine-learning models on student spending, as a Streamlit dashboard.

Run:  streamlit run app.py
"""
import warnings

import pandas as pd
import streamlit as st

from chillar import data, models, plots

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Chillar ML", page_icon="🪙", layout="wide")


@st.cache_data(show_spinner="Generating data and training the five models…")
def load(seed, days, budget):
    df = data.generate(days, seed)
    dd = data.daily(df, budget, days)
    return df, dd, models.run_all(df, dd)


@st.cache_data(show_spinner="Re-running on five datasets…")
def stability(days, budget):
    return models.compare_seeds(days=days, budget=budget)


def money(v): return f"₹{v:,.0f}"


def show(fig):
    st.pyplot(fig)
    plots.plt.close(fig)


def vs_baseline(label, model, base, lower_is_better=False, fmt=lambda v: f"{v:.1%}"):
    better = model < base if lower_is_better else model > base
    st.metric(label, fmt(model), f"{'better' if better else 'worse'} than baseline ({fmt(base)})",
              delta_color="normal" if better else "inverse")


with st.sidebar:
    st.title("Chillar ML")
    st.caption("Machine learning on a student's daily spending.")
    seed = st.number_input("Dataset seed", 0, 9999, 7, help="Each seed generates a different but similarly patterned dataset.")
    days = st.slider("Days of history", 60, 240, 120, 10)
    budget = st.slider("Daily budget (₹)", 80, 300, 150, 10, help="A day 'goes over' when spending exceeds this.")

df, dd, R = load(seed, days, budget)
clf, fc, rk, an, wk = R["classifier"], R["forecaster"], R["risk"], R["anomalies"], R["weeks"]

tabs = st.tabs(["Data", "1 · Classifier", "2 · Forecaster", "3 · Overspend risk", "4 · Anomalies", "5 · Week types", "Stability"])

with tabs[0]:
    st.header("The dataset")
    st.write("Synthetic spending for one student, generated with patterns a good model should find: Fridays and Saturdays "
             "cost more, the days after the allowance arrives are splurgier, the month end is leaner, and three one-off big "
             "purchases are hidden in the data as true anomalies.")
    c = st.columns(4)
    c[0].metric("Expenses", len(df)); c[1].metric("Days", len(dd)); c[2].metric("Average per day", money(dd["spent"].mean()))
    c[3].metric("Days over budget", f"{dd['over'].mean():.0%}")
    show(plots.eda(dd, budget))
    show(plots.category_totals(df))
    st.dataframe(df.assign(date=df["date"].dt.date), hide_index=True, width="stretch", height=260)

with tabs[1]:
    st.header("1 · Category classifier")
    st.write("**Concept:** supervised text classification. Each note (e.g. “uber to airport”) is turned into TF-IDF features "
             "from words and character n-grams, and a multinomial **Logistic Regression** predicts one of eight categories.")
    note = st.text_input("Try it: type something you might buy", "dominos with roommates")
    for cat, p in models.predict_category(clf["model"], note):
        st.progress(p, text=f"{cat} · {p:.0%}")
    c = st.columns(4)
    with c[0]: vs_baseline("Test accuracy", clf["accuracy"], clf["baseline_accuracy"])
    c[1].metric("Cross-validated accuracy", f"{clf['cv_mean']:.1%}", f"± {clf['cv_std']:.1%} over {clf['cv_folds']} folds", delta_color="off")
    c[2].metric("Macro F1", f"{clf['macro_f1']:.2f}"); c[3].metric("Labelled notes", clf["n_samples"])
    st.caption(f"Split: 75/25 grouped by phrase, so every test note is a phrase the model never saw. Baseline: {clf['baseline']}.")
    a, b = st.columns(2)
    with a: show(plots.confusion(clf))
    with b: show(plots.per_class_f1(clf))

with tabs[2]:
    st.header("2 · Spending forecaster")
    st.write("**Concept:** supervised regression. A **Random Forest** predicts tomorrow's spend from calendar features "
             "(weekday, days since payday) and lag features (yesterday, same day last week, 7- and 14-day averages). "
             "Forecasts are made recursively: each predicted day feeds the next.")
    c = st.columns(3)
    with c[0]: vs_baseline("MAE per day", fc["mae"], fc["baseline_mae"], True, money)
    with c[1]: vs_baseline("RMSE per day", fc["rmse"], fc["baseline_rmse"], True, money)
    c[2].metric("Next 14 days (predicted)", money(R["forecast"]["predicted"].sum()))
    st.caption(f"Split: time-ordered, train {fc['n_train']} days, test the last {fc['n_test']}. Baseline: {fc['baseline']}. "
               f"One-off spikes above {money(fc['cap'])} are capped in training.")
    show(plots.forecast_ahead(dd, R["forecast"], budget))
    show(plots.forecast_test(fc))

with tabs[3]:
    st.header("3 · Overspend risk")
    st.write("**Concept:** supervised binary classification. A class-balanced **Logistic Regression** on standardised "
             "features estimates the probability that a day ends over budget. ROC AUC is the main metric because it "
             "doesn't depend on a threshold.")
    c = st.columns(3)
    c[0].metric("ROC AUC", f"{rk['auc']:.2f}", "0.50 = coin flip", delta_color="off")
    with c[1]: vs_baseline("Accuracy at 0.5", rk["accuracy"], rk["baseline_accuracy"])
    c[2].metric("Days over budget", f"{rk['base_rate']:.0%}")
    a, b = st.columns([1, 2])
    with a: show(plots.roc(rk))
    with b: show(plots.importance(fc, rk))
    st.subheader("Risk for the next 7 days")
    ra = R["risk_ahead"].assign(day=lambda x: x.index.strftime("%a %d %b"))
    st.bar_chart(ra.set_index("day")["risk"], y_label="Probability of going over", color="#6B5BB8")

with tabs[4]:
    st.header("4 · Anomaly detector")
    st.write("**Concept:** unsupervised anomaly detection. An **Isolation Forest** isolates points with random splits; "
             "unusual expenses need fewer splits. Features: log amount, amount ÷ category median, weekend, hour.")
    st.metric("Flagged expenses", f"{int(an['anomaly'].sum())} of {len(an)}")
    show(plots.anomalies(an))
    flagged = an[an["anomaly"]].sort_values("score", ascending=False)
    st.dataframe(flagged.assign(date=flagged["date"].dt.date)[["date", "note", "category", "amount", "reason", "score"]],
                 hide_index=True, width="stretch")

with tabs[5]:
    st.header("5 · Week types")
    st.write("**Concept:** unsupervised clustering. Each week is described by its share of spending per category plus its "
             "total, and **K-Means** (k = 3) groups similar weeks. Clusters are named from their centres.")
    c = st.columns(3)
    c[0].metric("Weeks", wk["n_weeks"]); c[1].metric("Silhouette", f"{wk['silhouette']:.2f}", "−1 to 1, higher = cleaner", delta_color="off")
    c[2].metric("Latest week", wk["weeks"]["type"].iloc[-1])
    show(plots.week_types(wk))
    st.dataframe(wk["weeks"], hide_index=True, width="stretch")

with tabs[6]:
    st.header("Stability across datasets")
    st.write("One lucky dataset can flatter a model, so the three supervised models are re-run on five independently "
             "generated datasets.")
    tab = stability(days, budget)
    show(plots.seeds(tab))
    show = tab[["seed", "clf_cv", "clf_base", "fc_mae", "fc_base_mae", "risk_auc"]].rename(columns={
        "clf_cv": "Classifier CV acc", "clf_base": "Majority baseline", "fc_mae": "Forecaster MAE", "fc_base_mae": "7-day avg MAE", "risk_auc": "Risk AUC"})
    st.dataframe(show.style.format({"Classifier CV acc": "{:.1%}", "Majority baseline": "{:.1%}", "Forecaster MAE": "₹{:.0f}",
                                    "7-day avg MAE": "₹{:.0f}", "Risk AUC": "{:.2f}"}), hide_index=True, width="stretch")

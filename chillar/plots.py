"""Matplotlib charts shared by the Dash app and the project report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CATEGORIES

FOREST, SAGE, AI, CORAL, INK, MUTED = "#2E5B3C", "#8DAE89", "#6B5BB8", "#D96F55", "#1D2A21", "#7A857C"
CAT_COLORS = dict(zip(CATEGORIES, ["#D9952B", "#4F86B5", "#C25F88", "#8664C2", "#2F9481", "#9A9330", "#D46A45", "#7F8A82"]))
CUR = "₹"  # the report sets this to "Rs." because Times New Roman has no rupee glyph


def style(serif=False):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "font.size": 10,
                         "figure.dpi": 150, **({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]} if serif else {})})


style()


def _fig(w, h, cols=1):
    fig, ax = plt.subplots(1, cols, figsize=(w, h))
    return fig, ax


def eda(days, budget=150):
    fig, ax = _fig(7.2, 2.6, 2)
    dow = days.groupby(days.index.dayofweek)["spent"].mean()
    ax[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color=[SAGE] * 4 + [FOREST] * 2 + [SAGE])
    ax[0].axhline(budget, ls="--", c=MUTED, lw=1)
    ax[0].set_title(f"Average spend by weekday\n(dashed line = {CUR}{budget} daily budget)", fontsize=10); ax[0].set_ylabel(f"{CUR} per day")
    part = pd.cut((days.index.day - 1) % 31, [-1, 4, 22, 31], labels=["Days 1–5\n(after payday)", "Days 6–23", "Days 24+\n(month end)"])
    grp = days.groupby(part, observed=False)["spent"].mean()
    ax[1].bar(grp.index.astype(str), grp.values, color=[CORAL, SAGE, FOREST]); ax[1].set_title("Average spend by time in month")
    fig.tight_layout(); return fig


def category_totals(df):
    fig, ax = _fig(7.2, 2.4)
    t = df.groupby("category")["amount"].sum().reindex(CATEGORIES).fillna(0)
    ax.barh(t.index[::-1], t.values[::-1], color=[CAT_COLORS[c] for c in t.index[::-1]])
    ax.set_xlabel(f"Total spent ({CUR})"); ax.set_title("Spending by category")
    fig.tight_layout(); return fig


def confusion(clf):
    cm = np.array(clf["confusion"]); labels = [l.split(" ")[0] for l in clf["labels"]]
    fig, ax = _fig(5.2, 4.4)
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=8)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); return fig


def per_class_f1(clf):
    f1 = clf["per_class_f1"]
    fig, ax = _fig(5.6, 2.6)
    ax.barh(list(f1)[::-1], list(f1.values())[::-1], color=AI); ax.set_xlim(0, 1); ax.set_xlabel("F1 score"); ax.set_title("Per-category F1 (test set)")
    fig.tight_layout(); return fig


def forecast_test(fc):
    t = fc["test"]
    fig, ax = _fig(7, 2.8)
    ax.plot(t.index, t["actual"], "o-", c=SAGE, label="Actual", lw=2)
    ax.plot(t.index, t["predicted"], "s--", c=AI, label=f"Random Forest (MAE {CUR}{fc['mae']:.0f})", lw=1.8)
    ax.plot(t.index, t["baseline"], ":", c=MUTED, label=f"7-day average (MAE {CUR}{fc['baseline_mae']:.0f})", lw=1.8)
    ax.set_ylabel(f"{CUR} spent"); ax.set_title(f"Forecaster on the {fc['n_test']} held-out days"); ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def forecast_ahead(days, fut, budget=150):
    hist = days["spent"].iloc[-21:]
    fig, ax = _fig(7, 2.6)
    ax.bar(hist.index, hist.values, color=[CORAL if v > budget else SAGE for v in hist.values], label="Actual")
    ax.bar(fut.index, fut["predicted"], color="#ECE8F8", edgecolor=AI, ls="--", label="Predicted")
    ax.axhline(budget, ls="--", c=MUTED, lw=1)
    ax.set_ylim(0, max(budget * 3, float(fut["predicted"].max()) * 1.2)); ax.set_ylabel(f"{CUR} per day")
    ax.set_title("Last 3 weeks and the next 2 weeks"); ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); return fig


def importance(fc, rk):
    fig, ax = _fig(7.4, 2.8, 2)
    imp = fc["importance"]
    ax[0].barh(list(imp)[::-1], list(imp.values())[::-1], color=AI); ax[0].set_title("Forecaster feature importance")
    dr = rk["drivers"]
    ax[1].barh(list(dr)[::-1], list(dr.values())[::-1], color=[CORAL if v > 0 else FOREST for v in list(dr.values())[::-1]])
    ax[1].axvline(0, c=INK, lw=.8); ax[1].set_title("Overspend-risk coefficients")
    fig.tight_layout(); return fig


def roc(rk):
    fig, ax = _fig(3.6, 3.2)
    ax.plot(rk["fpr"], rk["tpr"], c=AI, lw=2, label=f"Logistic regression (AUC {rk['auc']:.2f})")
    ax.plot([0, 1], [0, 1], ":", c=MUTED, label="Coin flip (0.50)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC on test days"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); return fig


def anomalies(an):
    fig, ax = _fig(7, 2.6)
    xs = an["category"].map({c: i for i, c in enumerate(CATEGORIES)}) + np.random.default_rng(0).uniform(-.25, .25, len(an))
    ok = ~an["anomaly"]
    ax.scatter(xs[ok], an["amount"][ok], s=10, c=SAGE, alpha=.7, label="Normal")
    ax.scatter(xs[~ok], an["amount"][~ok], s=36, c=CORAL, edgecolor=INK, lw=.5, label="Flagged by Isolation Forest")
    ax.set_yscale("log"); ax.set_xticks(range(len(CATEGORIES)), [c.split(" ")[0] for c in CATEGORIES]); ax.set_ylabel(f"Amount ({CUR}, log)")
    ax.set_title("Anomaly detection on all expenses"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); return fig


def week_types(wk):
    g = wk["groups"]
    fig, ax = _fig(6.6, 2.5)
    for i, row in g.iterrows():
        left = 0
        for c in CATEGORIES:
            if row[c] > .03:
                ax.barh(i, row[c], left=left, color=CAT_COLORS[c], label=c if c not in ax.get_legend_handles_labels()[1] else None)
                left += row[c]
    ax.set_yticks(range(len(g)), [f"{r['name']}\n({r['weeks']} wk, avg {CUR}{r['avg_total']:.0f})" for _, r in g.iterrows()])
    ax.set_xlabel("Share of weekly spending"); ax.set_title("K-Means week types (cluster centres)")
    ax.legend(frameon=False, fontsize=7, ncol=4, bbox_to_anchor=(0.5, -0.32), loc="upper center")
    fig.tight_layout(); return fig


def seeds(tab):
    fig, ax = _fig(7.4, 2.4, 3)
    x = np.arange(len(tab))
    ax[0].bar(x - .18, tab["clf_cv"], .36, color=AI, label="Model (CV)"); ax[0].bar(x + .18, tab["clf_base"], .36, color=MUTED, label="Majority class")
    ax[0].set_title("Classifier accuracy"); ax[0].set_ylim(0, 1)
    ax[1].bar(x - .18, tab["fc_mae"], .36, color=AI, label="Random Forest"); ax[1].bar(x + .18, tab["fc_base_mae"], .36, color=MUTED, label="7-day average")
    ax[1].set_title(f"Forecaster MAE ({CUR}, lower better)")
    ax[2].bar(x, tab["risk_auc"], .5, color=AI); ax[2].axhline(.5, ls=":", c=MUTED); ax[2].set_ylim(0, 1); ax[2].set_title("Risk model ROC AUC")
    for a in ax: a.set_xticks(x, [f"s{s}" for s in tab["seed"]], fontsize=8)
    ax[0].legend(frameon=False, fontsize=7); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); return fig

"""Chillar ML: five machine-learning models on student spending, as a Dash dashboard (pure Python).

Run locally:  python app.py   → http://127.0.0.1:8050
On Vercel, `app` (the Flask server inside Dash) is the entry point.
"""
import base64
import io
import os
import warnings
from functools import lru_cache

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")  # serverless file systems are read-only except /tmp
warnings.filterwarnings("ignore")

from dash import Dash, Input, Output, dash_table, dcc, html  # noqa: E402

from chillar import data, models, plots  # noqa: E402

FOREST, INK, MUTED, BG, PANEL, LINE, GOOD, BAD = "#2E5B3C", "#1D2A21", "#6B7A70", "#F6F3EA", "#FFFDF7", "#E3DECD", "#2E7D4F", "#C0503A"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


# ---------- cached computation: each model trains once per setting, and only when its tab is opened ----------
@lru_cache(maxsize=16)
def dataset(seed, days, budget):
    df = data.generate(days, seed)
    return df, data.daily(df, budget, days)


@lru_cache(maxsize=16)
def classifier(seed, days):
    return models.train_classifier(dataset(seed, days, 150)[0])


@lru_cache(maxsize=16)
def forecaster(seed, days, budget):
    dd = dataset(seed, days, budget)[1]
    fc = models.train_forecaster(dd)
    return fc, models.forecast(fc, dd)


@lru_cache(maxsize=16)
def risk(seed, days, budget):
    dd = dataset(seed, days, budget)[1]
    rk = models.train_risk(dd)
    return rk, models.risk_ahead(rk, forecaster(seed, days, budget)[0], dd)


@lru_cache(maxsize=16)
def anomalies(seed, days):
    return models.detect_anomalies(dataset(seed, days, 150)[0])


@lru_cache(maxsize=16)
def weeks(seed, days):
    return models.week_types(dataset(seed, days, 150)[0])


PRECOMPUTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chillar", "stability_{}_{}.json")


@lru_cache(maxsize=4)
def stability(days, budget):
    """Re-running 3 models on 5 datasets takes ~40 s on a small serverless CPU, so the default settings ship precomputed
    (regenerate with models.compare_seeds(...).to_json(...)); any other setting is computed live."""
    path = PRECOMPUTED.format(days, budget)
    if os.path.exists(path):
        import pandas as pd
        return pd.read_json(path)
    return models.compare_seeds(days=days, budget=budget)


# ---------- small building blocks ----------
def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plots.plt.close(fig)
    return html.Img(src="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    style=dict(width="100%", maxWidth="900px", background="white", borderRadius="12px", border=f"1px solid {LINE}", margin="8px 0"))


def money(v): return f"₹{v:,.0f}"


def metric(label, value, note=None, good=None):
    colour = MUTED if good is None else GOOD if good else BAD
    return html.Div([html.Div(label, style=dict(fontSize="13px", color=MUTED)), html.Div(value, style=dict(fontSize="28px", fontWeight=600, color=INK)),
                     html.Div(note or "", style=dict(fontSize="12.5px", color=colour, minHeight="16px"))],
                    style=dict(flex="1 1 160px", background=PANEL, border=f"1px solid {LINE}", borderRadius="12px", padding="12px 14px"))


def vs(label, model, base, lower_is_better=False, fmt=lambda v: f"{v:.1%}"):
    better = model < base if lower_is_better else model > base
    return metric(label, fmt(model), f"{'better' if better else 'worse'} than baseline ({fmt(base)})", better)


def row(*children): return html.Div(list(children), style=dict(display="flex", gap="12px", flexWrap="wrap", margin="12px 0"))
def concept(text): return dcc.Markdown(text, style=dict(fontSize="15px", lineHeight="1.6", maxWidth="900px"))
def note(text): return html.P(text, style=dict(fontSize="13px", color=MUTED, maxWidth="900px"))
def side_by_side(a, b): return html.Div([html.Div(a, style=dict(flex="1 1 360px")), html.Div(b, style=dict(flex="1 1 360px"))], style=dict(display="flex", gap="12px", flexWrap="wrap"))


def table(df, height=None):
    return dash_table.DataTable(df.to_dict("records"), [{"name": c, "id": c} for c in df.columns], page_size=12,
                                style_table=dict(overflowX="auto", maxWidth="900px", **({"maxHeight": height, "overflowY": "auto"} if height else {})),
                                style_cell=dict(fontFamily=FONT, fontSize="13px", padding="6px 8px", textAlign="left", background=PANEL),
                                style_header=dict(fontWeight=600, background="#EFEBDD"))


# ---------- tabs ----------
def tab_data(seed, days, budget):
    df, dd = dataset(seed, days, budget)
    shown = df.assign(date=df["date"].dt.strftime("%Y-%m-%d"))
    return [html.H2("The dataset"),
            concept("Synthetic spending for one student, generated with patterns a good model should find: Fridays and Saturdays cost more, "
                    "the days after the allowance arrives are splurgier, the month end is leaner, and three one-off big purchases are hidden "
                    "in the data as true anomalies."),
            row(metric("Expenses", f"{len(df)}"), metric("Days", f"{len(dd)}"), metric("Average per day", money(dd["spent"].mean())),
                metric("Days over budget", f"{dd['over'].mean():.0%}")),
            img(plots.eda(dd, budget)), img(plots.category_totals(df)), table(shown, "340px")]


def tab_classifier(seed, days, budget):
    c = classifier(seed, days)
    return [html.H2("1 · Category classifier"),
            concept("**Concept:** supervised text classification. Each note (e.g. “uber to airport”) is turned into TF-IDF features from words and "
                    "character n-grams, and a multinomial **Logistic Regression** predicts one of eight categories."),
            html.Label("Try it: type something you might buy", style=dict(fontSize="14px", fontWeight=600)),
            dcc.Input(id="note", value="dominos with roommates", type="text", debounce=True,
                      style=dict(width="100%", maxWidth="600px", padding="10px 12px", fontSize="15px", borderRadius="10px", border=f"1px solid {LINE}", display="block", margin="6px 0")),
            html.Div(id="prediction"),
            row(vs("Test accuracy", c["accuracy"], c["baseline_accuracy"]),
                metric("Cross-validated accuracy", f"{c['cv_mean']:.1%}", f"± {c['cv_std']:.1%} over {c['cv_folds']} folds"),
                metric("Macro F1", f"{c['macro_f1']:.2f}"), metric("Labelled notes", f"{c['n_samples']}")),
            note(f"Split: 75/25 grouped by phrase, so every test note is a phrase the model never saw. Baseline: {c['baseline']}."),
            side_by_side(img(plots.confusion(c)), img(plots.per_class_f1(c)))]


def tab_forecaster(seed, days, budget):
    fc, fut = forecaster(seed, days, budget)
    dd = dataset(seed, days, budget)[1]
    return [html.H2("2 · Spending forecaster"),
            concept("**Concept:** supervised regression. A **Random Forest** predicts a day's spend from calendar features (weekday, days since payday) "
                    "and lag features (yesterday, same day last week, 7- and 14-day averages). Forecasts are recursive: each predicted day feeds the next."),
            row(vs("MAE per day", fc["mae"], fc["baseline_mae"], True, money), vs("RMSE per day", fc["rmse"], fc["baseline_rmse"], True, money),
                metric("Next 14 days (predicted)", money(fut["predicted"].sum()))),
            note(f"Split: time-ordered, train {fc['n_train']} days, test the last {fc['n_test']}. Baseline: {fc['baseline']}. "
                 f"One-off spikes above {money(fc['cap'])} are capped in training."),
            img(plots.forecast_ahead(dd, fut, budget)), img(plots.forecast_test(fc))]


def tab_risk(seed, days, budget):
    rk, ahead = risk(seed, days, budget)
    fc = forecaster(seed, days, budget)[0]
    ahead = ahead.assign(day=ahead.index.strftime("%a %d %b"), risk=lambda x: (x["risk"] * 100).round(0).astype(int).astype(str) + "%")[["day", "risk"]]
    return [html.H2("3 · Overspend risk"),
            concept("**Concept:** supervised binary classification. A class-balanced **Logistic Regression** on standardised features estimates the "
                    "probability that a day ends over budget. ROC AUC is the main metric because it doesn't depend on a threshold."),
            row(metric("ROC AUC", f"{rk['auc']:.2f}", "0.50 = coin flip"), vs("Accuracy at 0.5", rk["accuracy"], rk["baseline_accuracy"]),
                metric("Days over budget", f"{rk['base_rate']:.0%}")),
            side_by_side(img(plots.roc(rk)), img(plots.importance(fc, rk))),
            html.H3("Risk for the next 7 days"), table(ahead.rename(columns={"day": "Day", "risk": "Chance of going over"}))]


def tab_anomalies(seed, days, budget):
    an = anomalies(seed, days)
    flagged = an[an["anomaly"]].sort_values("score", ascending=False)
    flagged = flagged.assign(date=flagged["date"].dt.strftime("%Y-%m-%d"), score=flagged["score"].round(3))[["date", "note", "category", "amount", "reason", "score"]]
    return [html.H2("4 · Anomaly detector"),
            concept("**Concept:** unsupervised anomaly detection. An **Isolation Forest** isolates points with random splits; unusual expenses need "
                    "fewer splits. Features: log amount, amount ÷ category median, weekend, hour."),
            row(metric("Flagged expenses", f"{len(flagged)} of {len(an)}")), img(plots.anomalies(an)), table(flagged)]


def tab_weeks(seed, days, budget):
    wk = weeks(seed, days)
    return [html.H2("5 · Week types"),
            concept("**Concept:** unsupervised clustering. Each week is described by its share of spending per category plus its total, and "
                    "**K-Means** (k = 3) groups similar weeks. Clusters are named from their centres."),
            row(metric("Weeks", f"{wk['n_weeks']}"), metric("Silhouette", f"{wk['silhouette']:.2f}", "−1 to 1, higher = cleaner"),
                metric("Latest week", wk["weeks"]["type"].iloc[-1])),
            img(plots.week_types(wk)), table(wk["weeks"].assign(total=wk["weeks"]["total"].round(0)))]


def tab_stability(seed, days, budget):
    tab = stability(days, budget)
    show = tab.assign(**{"Classifier CV acc": (tab["clf_cv"] * 100).round(1).astype(str) + "%", "Majority baseline": (tab["clf_base"] * 100).round(1).astype(str) + "%",
                         "Forecaster MAE": tab["fc_mae"].map(money), "7-day avg MAE": tab["fc_base_mae"].map(money), "Risk AUC": tab["risk_auc"].round(2)})
    return [html.H2("Stability across datasets"),
            concept("One lucky dataset can flatter a model, so the three supervised models are re-run on five independently generated datasets."),
            *([] if os.path.exists(PRECOMPUTED.format(days, budget)) else [note("Computed live for these settings, which can take up to a minute on the hosted version.")]),
            img(plots.seeds(tab)), table(show[["seed", "Classifier CV acc", "Majority baseline", "Forecaster MAE", "7-day avg MAE", "Risk AUC"]])]


TABS = [("data", "Data", tab_data), ("clf", "1 · Classifier", tab_classifier), ("fc", "2 · Forecaster", tab_forecaster),
        ("risk", "3 · Overspend risk", tab_risk), ("anom", "4 · Anomalies", tab_anomalies), ("weeks", "5 · Week types", tab_weeks),
        ("stab", "Stability", tab_stability)]

# ---------- layout ----------
dash_app = Dash(__name__, title="Chillar ML", suppress_callback_exceptions=True)
app = dash_app.server  # the Flask app Vercel serves


def control(label, comp, help_text=None):
    return html.Div([html.Label(label, style=dict(fontSize="14px", fontWeight=600)), comp] +
                    ([html.Div(help_text, style=dict(fontSize="12px", color=MUTED))] if help_text else []), style=dict(marginBottom="22px"))


sidebar = html.Div([
    html.H1("Chillar ML", style=dict(fontSize="24px", margin="0 0 4px")),
    html.P("Machine learning on a student's daily spending.", style=dict(fontSize="13px", color=MUTED, marginTop=0)),
    html.Hr(style=dict(border=0, borderTop=f"1px solid {LINE}", margin="18px 0")),
    control("Dataset seed", dcc.Input(id="seed", type="number", value=7, min=0, max=9999, step=1, debounce=True,
                                      style=dict(width="100%", padding="8px", borderRadius="8px", border=f"1px solid {LINE}")),
            "Each seed generates a different but similarly patterned dataset."),
    control("Days of history", dcc.Slider(id="days", min=60, max=240, step=10, value=120, marks={60: "60", 120: "120", 180: "180", 240: "240"},
                                          tooltip=dict(placement="bottom", always_visible=False))),
    control("Daily budget (₹)", dcc.Slider(id="budget", min=80, max=300, step=10, value=150, marks={80: "80", 150: "150", 300: "300"},
                                           tooltip=dict(placement="bottom", always_visible=False)), "A day 'goes over' when spending exceeds this."),
], style=dict(width="250px", flex="0 0 250px", padding="28px 22px", background="#EFEBDD", minHeight="100vh", boxSizing="border-box"))

main = html.Div([
    dcc.Tabs(id="tab", value="data", children=[dcc.Tab(label=label, value=key) for key, label, _ in TABS],
             style=dict(fontSize="14px"), colors=dict(border=LINE, primary=FOREST, background=BG)),
    dcc.Loading(html.Div(id="content", style=dict(paddingTop="8px")), type="dot", color=FOREST),
], style=dict(flex="1 1 0%", padding="20px 32px 60px", minWidth="320px"))

dash_app.layout = html.Div([sidebar, main], style=dict(display="flex", alignItems="stretch", fontFamily=FONT, color=INK, background=BG, minHeight="100vh"))


def settings(seed, days, budget):
    seed = int(seed) if seed not in (None, "") else 7
    return max(0, min(9999, seed)), int(days or 120), int(budget or 150)


@dash_app.callback(Output("content", "children"), Input("tab", "value"), Input("seed", "value"), Input("days", "value"), Input("budget", "value"))
def show_tab(tab, seed, days, budget):
    fn = next(f for key, _, f in TABS if key == tab)
    return fn(*settings(seed, days, budget))


@dash_app.callback(Output("prediction", "children"), Input("note", "value"), Input("seed", "value"), Input("days", "value"))
def predict(text, seed, days):
    seed, days, _ = settings(seed, days, 150)
    guesses = models.predict_category(classifier(seed, days)["model"], text or "")
    if not guesses: return note("Type a note to see the model's guess.")
    bar = lambda cat, p: html.Div([html.Div(f"{cat} · {p:.0%}", style=dict(fontSize="14px", marginBottom="3px")),
                                   html.Div(html.Div(style=dict(width=f"{p * 100:.0f}%", height="8px", background=FOREST, borderRadius="4px")),
                                            style=dict(background=LINE, borderRadius="4px", maxWidth="600px"))], style=dict(margin="8px 0"))
    return html.Div([bar(c, p) for c, p in guesses], style=dict(margin="6px 0 14px"))


if __name__ == "__main__":
    dash_app.run(debug=False, port=8050)

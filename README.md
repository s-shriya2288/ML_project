# Chillar ML

📄 **Project report (21MCI002F):** [read the PDF](ML_Project_Report_Chillar_ML.pdf) · [Word version](report/ML_Project_Report_Chillar_ML.docx)

Five machine-learning models on a student's daily spending, written entirely in Python and shown in a Streamlit dashboard.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The sidebar sets the dataset seed, the number of days and the daily budget (₹150 by default). Each model has its own tab.

## The ML concepts

| # | Concept | Model | Evaluation | Baseline |
|---|---------|-------|------------|----------|
| 1 | Supervised **text classification**: the category from a note ("uber to airport" → Transport) | TF-IDF (word 1–2 grams + char 2–4 grams) → Logistic Regression | 75/25 split grouped by phrase, so test notes are unseen; 5-fold GroupKFold CV; accuracy, macro F1, confusion matrix | Always predict the most common category |
| 2 | Supervised **regression**: daily spend | Random Forest on calendar + lag features | Time-ordered split (last 21 days held out); MAE, RMSE, feature importance | Mean of the previous 7 days |
| 3 | Supervised **binary classification**: will a day go over budget? | Class-balanced Logistic Regression | Time-ordered split (last 28 days held out); ROC AUC, accuracy, coefficients | Always predict "not over" |
| 4 | Unsupervised **anomaly detection**: unusual expenses | Isolation Forest | Flags ~3% of expenses, each with a reason | n/a |
| 5 | Unsupervised **clustering**: week types | K-Means (k = 3) on each week's category mix | Silhouette score | n/a |

Results on five independently generated datasets (mean):

- **Classifier:** 70% cross-validated accuracy vs 45% for the baseline.
- **Forecaster:** MAE ₹68 vs ₹81 for the 7-day average; better on 4 of 5 seeds.
- **Risk model:** ROC AUC 0.84.

## The data

`chillar/data.py` generates 120 days of synthetic expenses (from 1 July 2026, so results are reproducible) with patterns the models should find:

- Fridays and Saturdays cost the most.
- Spending is 35% higher in the 5 days after the allowance arrives, and 20% lower at month end.
- There is log-normal day-to-day noise, plus some no-spend days.
- Weekends shift spending toward entertainment and shopping.
- Three one-off big purchases act as true anomalies.

## Files

- `app.py`: the Streamlit dashboard
- `chillar/data.py`: dataset generator and daily table
- `chillar/models.py`: the five models, their evaluation and the seed comparison
- `chillar/plots.py`: matplotlib charts shared by the dashboard and the report
- `report/`: the project report and the scripts that rebuild it

## Rebuild the report

```bash
.venv/bin/pip install -r report/requirements.txt
.venv/bin/python report/make_figures.py   # re-runs the models → report/metrics.json + report/figures/
.venv/bin/python report/build_report.py   # → report/ML_Project_Report_Chillar_ML.docx
```

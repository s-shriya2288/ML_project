# Chillar ML

A pocket-money tracker for students with a fixed virtual daily allowance (₹150 by default), a savings jar, and five machine-learning models that learn from your own spending. It only tracks virtual money: no bank links and no payments.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open http://127.0.0.1:5050. On the first screen, keep "Load 4 months of example spending" ticked so the models have data to train on straight away.

### Deploy to Vercel

The server is stateless, so it runs as a Vercel Python function with no setup: import this repo in Vercel (or run `npx vercel --prod`). Vercel detects the Flask `app` in `app.py` and installs `requirements.txt`.

### Where your data lives

Each visitor's data is kept in their own browser (`localStorage`) and sent with every request. The server applies the money rules, trains the models and sends the result back, but never stores anything. Visitors never see each other's data. Clearing browser data erases it.

## Android app and PhonePe import

`android/` is a small native Android app (plain Java, no libraries) that shows the live site and adds one native power: reading UPI debit SMS. Build it with `cd android && ./gradlew assembleDebug`, or download `/ChillarML.apk` from the deployed site.

There are two ways to bring in real spending (Expenses → Import from PhonePe & UPI):

- **UPI SMS** (Android app only): with your permission, `UpiSms.java` reads bank debit alerts (SBI, HDFC, ICICI, Axis, Kotak formats and card spends) on the phone, skipping OTPs, requests, refunds and money received. Only the amount, date, payee and reference are sent to the server; the SMS text never leaves the phone. It works for PhonePe, GPay or any UPI app.
- **PhonePe statement PDF**: PhonePe → History → Download statement, then upload. `importer.py` reads the debits and ignores credits. Password-protected PDFs are supported.

Imported payments are auto-categorised by the classifier from the payee name. References already imported, and manual entries with the same date and amount, are skipped. Only payments since your start date count, because earlier days have no allowance.

## How the money works

- Each day gets the daily allowance plus anything carried into it.
- At the end of a day, any leftover waits for you to choose: **Save it** (into the jar), **Carry it forward** (to the next day) or **Decide later**. Nothing moves on its own.
- Spending more than the day's balance is recorded as overspend and comes out of the monthly buffer (monthly allowance minus all the daily allowances). It is never counted twice.
- Goals are filled from free jar money. You also earn XP, levels, streaks and badges.

## The models (`ml.py`)

| # | Task | Model | How it's evaluated | Baseline |
|---|------|-------|--------------------|----------|
| 1 | Guess the category from the note ("uber to airport" → Transport) | TF-IDF (word 1–2 grams + character 2–4 grams) → Logistic Regression | 75/25 split grouped by phrase, so test notes are never seen in training, plus 5-fold GroupKFold cross-validation, accuracy, macro F1 and a confusion matrix | Always guess the most common category |
| 2 | Forecast daily spend for the rest of the month | Random Forest regressor on day of week, days since payday, day of month and lags (yesterday, same day last week, 7- and 14-day averages) | Time-ordered split (last 21 days held out), MAE and RMSE, feature importance | Average of the previous 7 days |
| 3 | Probability that today goes over budget | Logistic Regression (class-balanced, standardised features) | Time-ordered split, ROC AUC and accuracy, coefficients as risk drivers | Always guess "not over" |
| 4 | Flag unusual expenses | Isolation Forest on log amount, amount ÷ category median, weekend and hour | Unsupervised; flags about 3% of expenses with a plain-language reason | None |
| 5 | Name your "week types" | K-Means (k = 3) on each week's category mix + log weekly total | Silhouette score | None |

Actions like adding an expense never train anything, so they respond instantly. AI Insights fits the prediction models in the background (about 0.2 s on 4 months of data), and the full held-out evaluation and cross-validation run only when you open ML Lab. Results are cached per visitor in memory. The **ML Lab** page shows each model's metrics next to its baseline. The **AI Insights** page shows the predictions.

### Results on the example data (seed 7)

- **Classifier:** about 64% accuracy on held-out phrases (baseline 51%), and 70% ± 8% in cross-validation. Across 5 seeds, cross-validated accuracy is 70–74% against baselines of 38–53%.
- **Forecaster:** MAE ₹109 a day vs ₹126 for the 7-day-average baseline. On other seeds it is roughly level with the baseline, because daily student spending is noisy. One-off spikes above 3× the median day are capped during training; the anomaly detector handles those.
- **Overspend risk:** ROC AUC 0.81 (0.71–0.82 across seeds). Weekends and the days just after payday are the strongest drivers.

### About the example data

`engine.sample_state` generates 120 days of synthetic spending with built-in patterns: Fridays and Saturdays cost more, the first days after payday are splurgier, the last week of the month is leaner, plus random noise and three one-off big purchases. These are what the models are expected to discover. Your own data replaces this when you choose **Start fresh** in Profile.

## Project report

`report/ML_Project_Report_Chillar_ML.docx` is the 21MCI002F project report (SRM template). Everything in it is regenerated from the code:

```bash
.venv/bin/pip install -r report/requirements.txt
.venv/bin/python report/make_figures.py        # re-runs all models on 5 seeds → report/metrics.json + report/figures/
.venv/bin/python report/build_report.py        # → report/ML_Project_Report_Chillar_ML.docx
```

`report/page_map.py` reads a PDF export of the report to fill in the table-of-contents page numbers (pass the resulting JSON to `build_report.py`).

## Files

- `app.py`: stateless Flask JSON API and page server
- `engine.py`: money rules, input validation and the example-data generator
- `ml.py`: the five models, evaluation and predictions
- `templates/index.html`: the single-page web UI (vanilla JS, hand-drawn SVG charts)
- `importer.py`: PhonePe statement parser and the shared import step
- `android/`: the Android app; `static/ChillarML.apk` is the built APK the site serves

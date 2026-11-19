"""Synthetic student-spending dataset.

Real day-by-day spending data for Indian students isn't publicly available, so this generator produces it with
patterns a good model should be able to find:
- weekday rhythm: Fridays and Saturdays cost the most
- payday effect: +35% in the 5 days after the allowance arrives, −20% in the last week of the month
- log-normal day-to-day noise and occasional no-spend days
- weekends shift spending toward entertainment and shopping
- three one-off big purchases that act as true anomalies
Dates are fixed (not "today") so results are reproducible on any day.
"""
import random
from datetime import date, timedelta

import pandas as pd

CATEGORIES = ["Food & snacks", "Transport", "Shopping", "Fun & outings", "Study", "Personal care", "Bills & recharges", "Other"]
START = date(2026, 7, 1)

NOTES = {
    "Food & snacks": ["masala chai and bun", "canteen samosa", "maggi at the tapri", "cold coffee", "momos with the gang", "vada pav",
                      "zomato biryani", "fruit bowl", "cutting chai x2", "paneer roll", "swiggy pizza", "dosa at the stall", "bubble tea",
                      "lunch thali", "egg roll"],
    "Transport": ["auto to college", "metro card top up", "rapido to mall", "bus fare", "shared auto", "uber home late", "ola to station",
                  "petrol for scooty"],
    "Shopping": ["thrift store tee", "phone cover", "earrings from sarojini", "hair clips", "new sneakers", "myntra kurta", "tote bag"],
    "Fun & outings": ["movie ticket", "bowling night", "arcade tokens", "open mic entry", "concert pass", "gaming cafe", "escape room"],
    "Study": ["photocopies", "notebook and pens", "lab manual", "printouts for assignment", "udemy course", "graph paper", "exam form fee"],
    "Personal care": ["haircut", "face wash", "lip balm", "deo", "shampoo", "nail cutter", "sunscreen"],
    "Bills & recharges": ["jio recharge", "spotify student", "netflix mobile", "airtel data pack", "wifi share", "icloud storage"],
    "Other": ["laundry", "birthday gift share", "donation box", "key duplicate", "umbrella", "plant for room"],
}
WEIGHT = [("Food & snacks", .48), ("Transport", .2), ("Shopping", .05), ("Fun & outings", .08), ("Study", .08), ("Personal care", .05),
          ("Bills & recharges", .02), ("Other", .04)]
SUFFIX = ["", "", "", " near hostel", " with friends", " after class", " x2", " (split)", " again"]
DOW_BASE = [95, 85, 100, 105, 185, 225, 150]  # Mon..Sun average spend in ₹


def generate(days=120, seed=7, payday=1):
    """One row per expense: date, amount (₹), category, note, hour."""
    R = random.Random(seed)
    rows = []
    for i in range(days):
        d = START + timedelta(days=i)
        weekend = d.weekday() >= 4
        if R.random() < (.03 if weekend else .09):
            continue  # a no-spend day
        since_pay = (d.day - payday) % 31
        target = DOW_BASE[d.weekday()] * (1.35 if since_pay < 5 else .8 if since_pay > 22 else 1) * R.lognormvariate(0, .3)
        n = 1 if target < 70 else 2 if target < 160 else 3
        shares = [R.random() + .4 for _ in range(n)]
        for k in range(n):
            if k == 0:
                cat = R.choice(["Food & snacks", "Food & snacks", "Transport"])
            else:
                x, cat = R.random(), "Food & snacks"
                for c, p in WEIGHT:
                    x -= p * (2.2 if weekend and c in ("Fun & outings", "Shopping") else 1)
                    if x < 0:
                        cat = c
                        break
            rows.append(dict(date=d, amount=max(10, round(target * shares[k] / sum(shares) / 5) * 5), category=cat,
                             note=R.choice(NOTES[cat]) + R.choice(SUFFIX), hour=min(23, 9 + k * 3 + R.randint(0, 2))))
    for _ in range(3):  # one-off big purchases: what the anomaly detector should catch
        d = START + timedelta(days=R.randint(3, days - 3))
        cat = R.choice(["Shopping", "Fun & outings", "Food & snacks"])
        note = {"Shopping": "new headphones", "Fun & outings": "fest pass", "Food & snacks": "birthday treat for everyone"}[cat]
        rows.append(dict(date=d, amount=R.choice([899, 1199, 1499]), category=cat, note=note, hour=18))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def daily(df, budget=150, days=None):
    """One row per calendar day (including no-spend days): total spent and whether it went over the daily budget."""
    end = df["date"].max() if days is None else pd.Timestamp(START) + pd.Timedelta(days=days - 1)
    idx = pd.date_range(pd.Timestamp(START), end, freq="D")
    out = df.groupby("date")["amount"].sum().reindex(idx, fill_value=0).rename("spent").to_frame()
    out.index.name = "date"
    out["over"] = (out["spent"] > budget).astype(int)
    return out

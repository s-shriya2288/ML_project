"""Chillar ML: the pocket-money jar with machine learning. Run `python app.py`, then open http://127.0.0.1:5050

The server is stateless so it can run on serverless hosts like Vercel: each visitor's data lives in their own browser
(localStorage) and is sent with every request. The server applies the money rules, trains the models and sends the
updated state back. Nobody's data is stored on the server or shared with other visitors."""
import os
import time
import warnings

from flask import Flask, abort, jsonify, request, send_from_directory

import engine
from engine import CATS, CAT_NAMES, Engine, add_days, r2, today, uid

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # a few years of expenses fits comfortably
PAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def snapshot(s):
    e = Engine(s)
    t = today()
    exps = sorted(s["expenses"], key=lambda x: (x["date"], x.get("ts", 0)), reverse=True)
    return dict(profile=s["profile"], sample=s.get("sample", False), today=e.day(t), month=e.month(e.cycle_of(t)), jar=e.jar(),
                pending=e.pending(), streaks=e.streaks(), xp=e.xp(), badges=e.badges(), goals=s["goals"], budgets=s["budgets"],
                expenses=[{**x, "catName": CAT_NAMES[x["cat"]]} for x in exps[:300]],
                moves=sorted(s["moves"] + s["jarlog"], key=lambda m: m["ts"], reverse=True)[:30],
                week=[e.day(d) for d in engine.days_in(add_days(t, -6), t)], cats=CATS)


def fail(msg): abort(400, msg)


# ---- actions: each takes (state, body) and changes state in place ----
def a_add(s, b):
    amt, d = r2(b.get("amount")), b.get("date") or today()
    if amt <= 0 or b.get("cat") not in CAT_NAMES or d > today(): fail("Check the amount, category and date")
    s["expenses"].append(dict(id=uid(), amount=amt, cat=b["cat"], date=d, note=str(b.get("note", ""))[:60], pay=str(b.get("pay", "UPI"))[:10], ts=time.time()))


def a_edit(s, b):
    x = next((q for q in s["expenses"] if q["id"] == b.get("id")), None) or fail("Expense not found")
    if r2(b.get("amount")) <= 0 or b.get("cat") not in CAT_NAMES or (b.get("date") or x["date"]) > today(): fail("Check the amount, category and date")
    x.update(amount=r2(b["amount"]), cat=b["cat"], date=b.get("date") or x["date"], note=str(b.get("note", ""))[:60], pay=str(b.get("pay", x.get("pay")))[:10])


def a_delete(s, b): s["expenses"] = [q for q in s["expenses"] if q["id"] != b.get("id")]


def a_decide(s, b):
    d, c, t = b.get("date"), b.get("choice"), today()
    if c not in ("save", "carry", "later") or not d or Engine(s).day(d)["unused"] <= 0: fail("Nothing to decide for that day")
    x = Engine(s).day(d)
    s["later"] = [z for z in s["later"] if z != d]
    if c == "later":
        s["later"].append(d)
        return
    mv = {"id": uid(), "ts": time.time(), "type": c, "from": d, "amount": x["unused"]}
    if c == "carry": mv["to"] = add_days(d, 1) if d == t or add_days(d, 1) > t else t
    s["moves"].append(mv)


def a_goal_add(s, b):
    if not str(b.get("name", "")).strip() or r2(b.get("target")) <= 0: fail("Name and target needed")
    s["goals"].append(dict(id=uid(), name=str(b["name"])[:40], target=r2(b["target"]), saved=0))


def a_goal_money(s, b):
    g = next((q for q in s["goals"] if q["id"] == b.get("id")), None) or fail("Goal not found")
    a = r2(b.get("amount"))
    lim = min(Engine(s).jar()["free"], g["target"] - g["saved"]) if a > 0 else -g["saved"]
    if a == 0 or (a > 0 and a > lim + .001) or (a < 0 and a < lim - .001): fail("That amount doesn't fit")
    g["saved"] = r2(g["saved"] + a)


def a_goal_delete(s, b): s["goals"] = [g for g in s["goals"] if g["id"] != b.get("id")]


def a_settings(s, b):
    if r2(b.get("monthly")) <= 0 or r2(b.get("daily")) <= 0: fail("Allowances must be above zero")
    s["profile"].update(name=str(b.get("name", ""))[:30], monthly=r2(b["monthly"]), daily=r2(b["daily"]))


ACTIONS = dict(add=a_add, edit=a_edit, delete=a_delete, decide=a_decide, goal_add=a_goal_add, goal_money=a_goal_money,
               goal_delete=a_goal_delete, settings=a_settings)


def body_state():
    b = request.get_json(silent=True) or {}
    s = b.get("state")
    if not isinstance(s, dict): fail("Missing state")
    try:
        return b, engine.normalize(s)
    except (KeyError, TypeError, ValueError, AttributeError):
        fail("That saved data looks damaged. Start fresh from Profile.")


@app.get("/")
def index(): return send_from_directory(PAGE_DIR, "index.html")  # plain file: the page is pure JS, no Jinja


@app.post("/api/setup")
def setup():
    b = request.get_json(silent=True) or {}
    try:
        prof = dict(name=str(b.get("name", ""))[:30], monthly=r2(b.get("monthly")), daily=r2(b.get("daily")),
                    payday=int(min(31, max(1, int(b.get("payday") or 1)))))
    except (TypeError, ValueError):
        fail("Check your allowances")
    if prof["monthly"] <= 0 or prof["daily"] <= 0: fail("Allowances must be above zero")
    s = engine.sample_state(prof) if b.get("sample") else engine.blank_state(prof)
    return jsonify(state=s, snap=snapshot(s))


@app.post("/api/reset")
def reset():
    b, s = body_state()
    prof = {k: s["profile"][k] for k in ("name", "monthly", "daily", "payday")}
    s = engine.sample_state(prof) if b.get("sample") else engine.blank_state(prof)
    return jsonify(state=s, snap=snapshot(s))


@app.post("/api/do/<action>")
def do(action):
    fn = ACTIONS.get(action) or abort(404)
    b, s = body_state()
    fn(s, b)
    s["updatedAt"] = time.time()
    return jsonify(state=s, snap=snapshot(s))


@app.post("/api/snapshot")
def snap():
    _, s = body_state()
    return jsonify(state=s, snap=snapshot(s))


# ---------- machine learning endpoints ----------
# scikit-learn and pandas are imported on first use, so the plain money endpoints start fast on a cold server.
def ml():
    import ml as m
    return m


@app.post("/api/suggest")
def suggest():
    b, s = body_state()
    return jsonify(ml().suggest(s, str(b.get("note", ""))[:80]))


@app.post("/api/insights")
def insights():
    _, s = body_state()
    return jsonify(ml().insights(s))


@app.post("/api/lab")
def lab():
    b, s = body_state()
    if b.get("retrain"): ml().forget(s)
    M = ml().models(s, evaluate=True)
    return jsonify(metrics=M["metrics"], trained_at=M["trained_at"], train_seconds=M["train_seconds"])


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def err(e): return jsonify(error=e.description if isinstance(e.description, str) else "Bad request"), e.code


if __name__ == "__main__":
    app.run(debug=False, port=5050)

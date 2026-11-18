"""Money rules for Chillar ML: a fixed virtual daily allowance, end-of-day save/carry choices, a savings jar,
goals, streaks and XP. Pure Python so the ML layer and the Flask API can both use it."""
import random
import time
import uuid
from datetime import date, timedelta

CATS = [("food", "Food & snacks"), ("transport", "Transport"), ("shopping", "Shopping"), ("entertainment", "Fun & outings"),
        ("education", "Study"), ("personal", "Personal care"), ("bills", "Bills & recharges"), ("other", "Other")]
CAT_NAMES = dict(CATS)
DEFAULT_BUDGETS = dict(food=1500, transport=600, shopping=700, entertainment=600, education=400, personal=300, bills=500, other=300)
TITLES = ["Coin Rookie", "Chillar Collector", "Piggy Pro", "Jar Keeper", "Budget Bestie", "Streak Sorcerer", "Savings Sensei",
          "Wealth Wizard", "Money Monk", "Chillar Legend"]


def uid(): return uuid.uuid4().hex[:10]
def today(): return date.today().isoformat()
def pd(s): return date.fromisoformat(s)
def add_days(s, n): return (pd(s) + timedelta(days=n)).isoformat()
def r2(n): return round(float(n or 0), 2)


def days_in(a, b):
    out, d, e = [], pd(a), pd(b)
    while d <= e:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def dim(y, m):
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return (date(ny, nm, 1) - timedelta(days=1)).day


# ---------- state ----------
def blank_state(profile=None):
    p = dict(name="", monthly=6000, payday=1, daily=150, startDate=today())
    p.update(profile or {})
    return dict(v=1, updatedAt=time.time(), profile=p, expenses=[], moves=[], jarlog=[], later=[], goals=[],
                budgets=dict(DEFAULT_BUDGETS), sample=False)


def normalize(s):
    """State arrives from the visitor's browser, so check its shape and drop anything malformed before using it."""
    p = s.get("profile") or {}
    prof = dict(name=str(p.get("name", ""))[:30], monthly=r2(p.get("monthly", 6000)), daily=r2(p.get("daily", 150)),
                payday=int(min(31, max(1, int(p.get("payday", 1))))), startDate=pd(str(p.get("startDate", today()))).isoformat())
    if prof["monthly"] <= 0 or prof["daily"] <= 0: raise ValueError("bad allowance")
    out = blank_state(prof)
    out.update(updatedAt=float(s.get("updatedAt") or time.time()), sample=bool(s.get("sample")),
               later=[pd(d).isoformat() for d in s.get("later", [])][:400])
    for e in s.get("expenses", [])[:20000]:
        if e.get("cat") in CAT_NAMES and r2(e.get("amount")) > 0:
            out["expenses"].append(dict(id=str(e.get("id") or uid())[:20], amount=r2(e["amount"]), cat=e["cat"], date=pd(e["date"]).isoformat(),
                                        note=str(e.get("note", ""))[:60], pay=str(e.get("pay", "UPI"))[:10], ts=float(e.get("ts") or 0),
                                        **({"ref": str(e["ref"])[:60], "src": str(e.get("src", ""))[:12]} if e.get("ref") else {})))
    for m in s.get("moves", [])[:20000]:
        if m.get("type") in ("save", "carry") and r2(m.get("amount")) > 0:
            mv = {"id": str(m.get("id") or uid())[:20], "ts": float(m.get("ts") or 0), "type": m["type"], "from": pd(m["from"]).isoformat(), "amount": r2(m["amount"])}
            if m["type"] == "carry": mv["to"] = pd(m["to"]).isoformat()
            out["moves"].append(mv)
    out["jarlog"] = [dict(id=str(j.get("id") or uid())[:20], ts=float(j.get("ts") or 0), type="withdraw", amount=r2(j["amount"]), note=str(j.get("note", ""))[:60])
                     for j in s.get("jarlog", [])[:5000] if r2(j.get("amount")) > 0]
    out["goals"] = [dict(id=str(g.get("id") or uid())[:20], name=str(g.get("name", "Goal"))[:40], target=r2(g["target"]), saved=max(0, r2(g.get("saved"))))
                    for g in s.get("goals", [])[:50] if r2(g.get("target")) > 0]
    out["budgets"].update({k: r2(v) for k, v in (s.get("budgets") or {}).items() if k in CAT_NAMES and r2(v) >= 0})
    return out


# ---------- ledger ----------
class Engine:
    """One day's ledger: allowance = daily amount + money carried in. left = allowance − saved/carried out − spent.
    A negative left is overspend; it comes out of the monthly buffer (monthly allowance − all the daily allowances)."""

    def __init__(self, state):
        self.s = state
        self.spent, self.carry_to, self.out, self.saved_from, self.carried_from = {}, {}, {}, {}, {}

        def add(m, k, v): m[k] = r2(m.get(k, 0) + v)
        for e in state["expenses"]:
            add(self.spent, e["date"], e["amount"])
        for mv in state["moves"]:
            add(self.out, mv["from"], mv["amount"])
            add(self.saved_from if mv["type"] == "save" else self.carried_from, mv["from"], mv["amount"])
            if mv["type"] == "carry": add(self.carry_to, mv["to"], mv["amount"])
        self._day = {}

    def day(self, d):
        if d in self._day: return self._day[d]
        tracked = d >= self.s["profile"]["startDate"]
        base = self.s["profile"]["daily"] if tracked else 0
        cin = self.carry_to.get(d, 0)
        allowance = r2(base + cin)
        spent, moved = self.spent.get(d, 0), self.out.get(d, 0)
        left = r2(allowance - moved - spent)
        x = dict(d=d, tracked=tracked, base=base, cin=cin, allowance=allowance, spent=spent, moved=moved,
                 saved=self.saved_from.get(d, 0), carried=self.carried_from.get(d, 0), left=left, over=max(0, -left), unused=max(0, left))
        self._day[d] = x
        return x

    def cycle_of(self, s):
        p = self.s["profile"]["payday"]
        d = pd(s)
        y, m = d.year, d.month
        sd = min(p, dim(y, m))
        if d.day < sd:
            m -= 1
            if m < 1: m, y = 12, y - 1
            sd = min(p, dim(y, m))
        start = date(y, m, sd)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        nxt = date(ny, nm, min(p, dim(ny, nm)))
        return dict(start=start.isoformat(), end=(nxt - timedelta(days=1)).isoformat(), days=(nxt - start).days,
                    label=f"{start:%B %Y}" if sd == 1 else f"{start.day} {start:%b} – {(nxt - timedelta(days=1)).day} {(nxt - timedelta(days=1)):%b}")

    def month(self, cyc):
        s, t = self.s, today()
        upto = min(cyc["end"], t)
        exps = [e for e in s["expenses"] if cyc["start"] <= e["date"] <= cyc["end"]]
        spent = r2(sum(e["amount"] for e in exps))
        saved = r2(sum(m["amount"] for m in s["moves"] if m["type"] == "save" and cyc["start"] <= m["from"] <= cyc["end"]))
        days = [self.day(d) for d in days_in(cyc["start"], upto)] if upto >= cyc["start"] else []
        planned = r2(sum(s["profile"]["daily"] for d in days_in(cyc["start"], cyc["end"]) if d >= s["profile"]["startDate"]))
        M = s["profile"]["monthly"]
        by_cat = {c: 0 for c, _ in CATS}
        for e in exps: by_cat[e["cat"]] = r2(by_cat.get(e["cat"], 0) + e["amount"])
        over = r2(sum(x["over"] for x in days))
        return dict(cyc=cyc, spent=spent, saved=saved, overspend=over, buffer=r2(M - planned), buffer_left=r2(M - planned - over),
                    remaining=r2(M - spent - saved), by_cat=by_cat, elapsed=len([x for x in days if x["tracked"]]))

    def jar(self):
        dep = r2(sum(m["amount"] for m in self.s["moves"] if m["type"] == "save"))
        wd = r2(sum(j["amount"] for j in self.s["jarlog"]))
        alloc = r2(sum(g["saved"] for g in self.s["goals"]))
        return dict(deposits=dep, balance=r2(dep - wd), allocated=alloc, free=r2(dep - wd - alloc))

    def pending(self):
        t = today()
        start = max(self.s["profile"]["startDate"], add_days(t, -60))
        return [{**self.day(d), "later": d in self.s["later"]} for d in (days_in(start, add_days(t, -1)) if start < t else [])
                if self.day(d)["unused"] > 0.009]

    def streaks(self):
        t, st = today(), self.s["profile"]["startDate"]
        within, d = 0, add_days(t, -1)
        while d >= st and self.day(d)["left"] >= 0:
            within += 1
            d = add_days(d, -1)
        best = run = 0
        for d in days_in(st, add_days(t, -1)) if st < t else []:
            run = run + 1 if self.day(d)["left"] >= 0 else 0
            best = max(best, run)
        return dict(within=within, best=best)

    def xp(self):
        t, s = today(), self.s
        xp = 0
        for d in days_in(s["profile"]["startDate"], add_days(t, -1)) if s["profile"]["startDate"] < t else []:
            x = self.day(d)
            xp += (10 if x["left"] >= 0 else 0) + (15 if x["spent"] == 0 else 0)
        xp += int(self.jar()["deposits"] // 10) + 100 * len([g for g in s["goals"] if g["saved"] >= g["target"]])
        xp += 20 * len([b for b in self.badges() if b["got"]])
        lvl = xp // 400 + 1
        return dict(xp=xp, level=lvl, into=xp % 400, per=400, title=TITLES[min(len(TITLES) - 1, lvl - 1)])

    def badges(self):
        j, st = self.jar(), self.streaks()
        t, start = today(), self.s["profile"]["startDate"]
        nospend = len([d for d in days_in(start, add_days(t, -1)) if start < t and self.day(d)["spent"] == 0])
        return [dict(name="Jar Opener", desc="Saved your first leftover", got=j["deposits"] > 0),
                dict(name="First ₹500 Saved", desc="₹500 in the jar, all time", got=j["deposits"] >= 500),
                dict(name="Seven-Day Saver", desc="7 days in a row within budget", got=st["best"] >= 7),
                dict(name="No-Spend Ninja", desc="3 no-spend days", got=nospend >= 3),
                dict(name="Goal Getter", desc="Hit a savings goal", got=any(g["saved"] >= g["target"] for g in self.s["goals"])),
                dict(name="Data Scientist", desc="Logged 100 expenses for the models", got=len(self.s["expenses"]) >= 100)]


# ---------- example data with real patterns for the models to find ----------
NOTES = {
    "food": ["masala chai and bun", "canteen samosa", "maggi at the tapri", "cold coffee", "momos with the gang", "vada pav", "zomato biryani",
             "fruit bowl", "cutting chai x2", "paneer roll", "swiggy pizza", "dosa at the stall", "bubble tea", "lunch thali", "egg roll"],
    "transport": ["auto to college", "metro card top up", "rapido to mall", "bus fare", "shared auto", "uber home late", "ola to station", "petrol for scooty"],
    "shopping": ["thrift store tee", "phone cover", "earrings from sarojini", "hair clips", "new sneakers", "myntra kurta", "tote bag"],
    "entertainment": ["movie ticket", "bowling night", "arcade tokens", "open mic entry", "concert pass", "gaming cafe", "escape room"],
    "education": ["photocopies", "notebook and pens", "lab manual", "printouts for assignment", "udemy course", "graph paper", "exam form fee"],
    "personal": ["haircut", "face wash", "lip balm", "deo", "shampoo", "nail cutter", "sunscreen"],
    "bills": ["jio recharge", "spotify student", "netflix mobile", "airtel data pack", "wifi share", "icloud storage"],
    "other": ["laundry", "birthday gift share", "donation box", "key duplicate", "umbrella", "plant for room"],
}
PRICE = dict(food=(15, 110), transport=(15, 80), shopping=(150, 450), entertainment=(120, 300), education=(20, 120),
             personal=(40, 180), bills=(59, 299), other=(30, 160))
WEIGHT = [("food", .48), ("transport", .2), ("shopping", .05), ("entertainment", .08), ("education", .08), ("personal", .05), ("bills", .02), ("other", .04)]
EXTRA = ["", "", "", " near hostel", " with friends", " after class", " x2", " (split)", " again"]


def sample_state(profile, days=120, seed=7):
    """Synthetic but patterned history: Fridays/Saturdays cost more, the first days after payday are splurgier,
    exam weeks push Study up, and a few one-off big spends act as anomalies."""
    t = today()
    R = random.Random(seed)
    s = blank_state({**profile, "startDate": add_days(t, -days)})
    pay = s["profile"]["payday"]

    dow_base = [95, 85, 100, 105, 185, 225, 150]  # Mon..Sun: Fridays and Saturdays are outing days

    def cat(weekend, first):
        if first: return R.choice(["food", "food", "transport"])
        x = R.random()
        for c, p in WEIGHT:
            x -= p * (2.2 if weekend and c in ("entertainment", "shopping") else 1)
            if x < 0: return c
        return "food"
    for d in days_in(s["profile"]["startDate"], t):
        dd = pd(d)
        wd = dd.weekday()
        weekend = wd >= 4
        since_pay = (dd.day - pay) % 31
        if d != t and R.random() < (.03 if weekend else .09): continue  # a no-spend day
        target = dow_base[wd] * (1.35 if since_pay < 5 else .8 if since_pay > 22 else 1) * R.lognormvariate(0, .3)
        if d == t: target = 80
        n = 1 if target < 70 else 2 if target < 160 else 3
        shares = [R.random() + .4 for _ in range(n)]
        for i in range(n):
            c = cat(weekend, i == 0)
            amt = max(10, round(target * shares[i] / sum(shares) / 5) * 5)
            s["expenses"].append(dict(id=uid(), amount=amt, cat=c, date=d, note=R.choice(NOTES[c]) + R.choice(EXTRA), pay=R.choice(["UPI", "UPI", "Cash", "Card"]),
                                      ts=time.time() - (n - i) * 1800 if d == t else time.mktime(dd.timetuple()) + (9 + i * 3 + R.random() * 2) * 3600))
    for _ in range(3):  # one-off big spends the anomaly detector should notice
        d = add_days(t, -R.randint(3, days - 3))
        c = R.choice(["shopping", "entertainment", "food"])
        s["expenses"].append(dict(id=uid(), amount=R.choice([899, 1199, 1499]), cat=c, date=d, note={"shopping": "new headphones", "entertainment": "fest pass", "food": "birthday treat for everyone"}[c],
                                  pay="UPI", ts=time.mktime(pd(d).timetuple()) + 18 * 3600))
    eng = Engine(s)
    for d in days_in(s["profile"]["startDate"], add_days(t, -2)):  # leave yesterday undecided for the closing prompt
        eng = Engine(s)
        x = eng.day(d)
        if x["unused"] <= 0: continue
        ts = time.mktime(pd(d).timetuple()) + 22 * 3600
        if R.random() < .74: s["moves"].append({"id": uid(), "ts": ts, "type": "save", "from": d, "amount": x["unused"]})
        else: s["moves"].append({"id": uid(), "ts": ts, "type": "carry", "from": d, "to": add_days(d, 1), "amount": x["unused"]})
    bal = Engine(s).jar()["balance"]
    s["goals"] = [dict(id=uid(), name="New phone", target=15000, saved=round(bal * .4 / 10) * 10),
                  dict(id=uid(), name="Goa trip with the squad", target=8000, saved=round(bal * .2 / 10) * 10)]
    s["sample"] = True
    return s

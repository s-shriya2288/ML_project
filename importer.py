"""Bring real UPI spending into Chillar.

Two sources feed one import step:
- UPI debit SMS, read and parsed on the phone by the Android app (only amount, date, payee and reference arrive here).
- A PhonePe statement PDF (PhonePe → History → Download statement), parsed below.

Each debit becomes an expense, categorised by the classifier from its payee. Credits are ignored, references already
imported are skipped, and a manual entry with the same date and amount is treated as the same payment."""
import io
import re
import time
from collections import Counter
from datetime import datetime

from engine import CAT_NAMES, pd, r2, today, uid

MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
DATE = re.compile(rf"\b({MONTHS})[a-z]* (\d{{1,2}}), (\d{{4}})\b")
TIME = re.compile(r"\b(\d{1,2}):(\d{2}) ?([AaPp][Mm])\b")
AMOUNT = re.compile(r"(?:₹|Rs\.?|INR|`)\s*([\d,]+(?:\.\d{1,2})?)")
TXN_ID = re.compile(r"Transaction ID[:\s]*([A-Z0-9]{8,})", re.I)
PAYEE = re.compile(r"(Paid to|Payment to|Sent to|Mobile recharged|Bill paid -|Received from|Paid -)\s*(.*?)(?=\s+(?:Transaction ID|UTR|DEBIT|CREDIT|Debited|Credited|Paid by)\b|$)", re.I | re.S)


def statement_text(data, password=""):
    from pypdf import PdfReader  # imported on use so the rest of the app starts fast
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(password or ""):
        raise ValueError("This statement is password-protected. Enter the password PhonePe gave you (often your registered mobile number).")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_phonepe(text):
    """Split the statement at each date and read one transaction from each chunk. Returns (debits, credits_skipped)."""
    starts = [m.start() for m in DATE.finditer(text)]
    out, credits = [], 0
    for a, b in zip(starts, starts[1:] + [len(text)]):
        chunk = " ".join(text[a:b].split())
        kind = re.search(r"\b(DEBIT|CREDIT)\b", chunk)
        amt = AMOUNT.search(chunk) or re.search(r"\b(?:DEBIT|CREDIT)\s+([\d,]+(?:\.\d{1,2})?)", chunk)
        if not kind or not amt: continue
        if kind.group(1) == "CREDIT":
            credits += 1
            continue
        mo, dd, yy = DATE.search(chunk).groups()
        d = datetime.strptime(f"{mo[:3]} {dd} {yy}", "%b %d %Y")
        tm = TIME.search(chunk)
        if tm:
            h = int(tm.group(1)) % 12 + (12 if tm.group(3).lower() == "pm" else 0)
            d = d.replace(hour=h, minute=int(tm.group(2)))
        who = PAYEE.search(chunk)
        payee = (who.group(1).lower().startswith("mobile") and "Mobile recharge " or "") + (who.group(2).strip(" -:") if who else "")
        ref = TXN_ID.search(chunk)
        out.append(dict(amount=r2(amt.group(1).replace(",", "")), date=d.date().isoformat(), ts=d.timestamp(),
                        payee=payee[:60] or "PhonePe payment", ref="pp:" + (ref.group(1) if ref else f"{d:%Y%m%d%H%M}-{amt.group(1)}")))
    return out, credits


def add_transactions(state, txns, source, suggest):
    """Turn debits into expenses. Returns counts for the summary toast."""
    seen = {e.get("ref") for e in state["expenses"] if e.get("ref")}
    manual = Counter((e["date"], r2(e["amount"])) for e in state["expenses"] if not e.get("ref"))  # each manual entry matches one import
    start, t = state["profile"]["startDate"], today()
    res = dict(added=0, duplicates=0, older=0, invalid=0, by_cat={})
    for x in txns[:5000]:
        try:
            amt, d = r2(x.get("amount")), pd(str(x.get("date"))).isoformat()
            ref = str(x.get("ref") or "")[:60]
        except (TypeError, ValueError):
            res["invalid"] += 1
            continue
        if amt <= 0 or d > t:
            res["invalid"] += 1
        elif d < start:
            res["older"] += 1  # days before you started have no allowance to count against
        elif ref and ref in seen:
            res["duplicates"] += 1
        elif manual[(d, amt)] > 0:
            manual[(d, amt)] -= 1
            res["duplicates"] += 1
        else:
            payee = " ".join(str(x.get("payee") or "UPI payment").split())[:60]
            guess = suggest(state, payee)
            cat = guess[0]["cat"] if guess else "other"
            state["expenses"].append(dict(id=uid(), amount=amt, cat=cat, date=d, note=payee, pay="UPI", ts=float(x.get("ts") or time.time()),
                                          ref=ref or None, src=source))
            seen.add(ref)
            res["added"] += 1
            res["by_cat"][CAT_NAMES[cat]] = res["by_cat"].get(CAT_NAMES[cat], 0) + 1
    return res

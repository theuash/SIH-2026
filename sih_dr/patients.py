"""Patient consent store + notify log. JSON files, atomic writes, stdlib only."""
import json
import os
import re
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PATIENTS = DATA / "patients.json"
LOG = DATA / "notify_log.jsonl"

PHONE_RE = re.compile(r"^\+?\d{10,15}$")


def normalize_phone(raw):
    p = re.sub(r"[\s\-()]", "", raw or "")
    if p.startswith("00"):
        p = "+" + p[2:]
    if p.isdigit() and len(p) == 10:
        p = "+91" + p  # default clinic country
    if not PHONE_RE.match(p):
        raise ValueError("enter a valid mobile number")
    return p


def _load(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def set_consent(phone, lang="en"):
    phone = normalize_phone(phone)
    db = _load(PATIENTS, {})
    db[phone] = {"lang": lang, "consent_at": int(time.time()), "opted_out": False}
    _save_atomic(PATIENTS, db)
    return phone


def opt_out(phone):
    phone = normalize_phone(phone)
    db = _load(PATIENTS, {})
    db.setdefault(phone, {}).update({"opted_out": True})
    _save_atomic(PATIENTS, db)
    return phone


def is_opted_out(phone):
    try:
        return bool(_load(PATIENTS, {}).get(normalize_phone(phone), {}).get("opted_out"))
    except ValueError:
        return False


def log_notify(phone, lang, text, status, msg_id=""):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": int(time.time()), "phone": phone, "lang": lang,
                            "status": status, "id": msg_id,
                            "preview": text[:80]}) + "\n")


def handle_reply(phone, text):
    """Returns auto-reply text or ''. STOP/HELP honored; anything else logged."""
    t = (text or "").strip().upper()
    if t in {"STOP", "UNSUBSCRIBE", "रोकें"}:
        opt_out(phone)
        return "You are opted out. Reply START to rejoin."
    if t in {"HELP", "START", "मदद"}:
        return "NetraDR: screening updates only. A doctor confirms every result."
    log_notify(phone, "?", text or "", "inbound")
    return ""

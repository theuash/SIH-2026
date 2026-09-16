"""WhatsApp notify layer. Provider-agnostic; stdlib HTTP only.

Primary: Baileys sidecar (wa-gateway/, QR pairing) at WA_GATEWAY_URL.
Fallback: demo-queued mode — logs + returns a local id so the UI/pipeline
never blocks when the gateway is down. Swap to Cloud API by implementing
the same three functions against graph.facebook.com.
"""
import json
import os
import time
import urllib.request

GATEWAY = os.environ.get("WA_GATEWAY_URL", "http://127.0.0.1:3001")

TEMPLATES = {
    "recapture": {
        "en": "NetraDR: your eye photo was unclear ({reason}). Please retake at the camp. Reply STOP to opt out.",
        "hi": "NetraDR: आपकी आंख की फोटो स्पष्ट नहीं ({reason})। कृपया कैंप में दोबारा लें। रोकने हेतु STOP लिखें।"},
    "result": {
        "en": "NetraDR screening done. Grade {grade} ({name}), confidence {conf}. A doctor will confirm. {next} Reply STOP to opt out.",
        "hi": "NetraDR जांच पूरी। ग्रेड {grade} ({name}), विश्वास {conf}। डॉक्टर पुष्टि करेंगे। {next} रोकने हेतु STOP लिखें।"},
}

NEXT_STEP = {
    "en": {True: "Please visit the referral doctor within 2 weeks.",
           False: "No referral needed. Repeat screening next year."},
    "hi": {True: "कृपया 2 सप्ताह में रेफरल डॉक्टर से मिलें।",
           False: "रेफरल आवश्यक नहीं। अगले वर्ष दोबारा जांच कराएं।"},
}


def render(kind, lang, **kw):
    lang = lang if lang in ("en", "hi") else "en"
    return TEMPLATES[kind][lang].format(**kw)


def _post(path, payload, timeout=8):
    req = urllib.request.Request(GATEWAY + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def get_status():
    try:
        return _post("/status", {}, timeout=4)
    except Exception:
        return {"status": "disconnected", "gateway": False}


def get_qr():
    try:
        return _post("/qr", {}, timeout=10)
    except Exception:
        return {"qr": None, "status": "disconnected", "gateway": False}


def send_text(phone, text, lang="en"):
    from . import patients
    if patients.is_opted_out(phone):
        return {"ok": False, "error": "opted-out"}
    try:
        r = _post("/send", {"to": phone, "text": text})
        msg_id = r.get("id", "wa-" + str(int(time.time())))
        patients.log_notify(phone, lang, text, "sent", msg_id)
        return {"ok": True, "id": msg_id}
    except Exception as e:  # gateway down -> honest queued-demo mode
        msg_id = "queued-" + str(int(time.time()))
        patients.log_notify(phone, lang, text, "queued-demo", msg_id)
        return {"ok": True, "id": msg_id, "demo": True, "note": str(e)[:100]}

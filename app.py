"""NetraDR backend — real M1→M4 screening + WhatsApp notify + M5 sim.
Run: python app.py  ->  http://127.0.0.1:8000
"""
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from sih_dr.m5_sim import runScreeningSim
from sih_dr import notify_whatsapp as wa
from sih_dr import patients

ROOT = Path(__file__).parent
app = FastAPI(title="NetraDR API")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/screen")
async def screen(image: UploadFile = File(...)):
    from sih_dr.run_pipeline import screen_image
    raw = await image.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "unreadable image"}, status_code=400)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = screen_image(img)
    g = r["grading"]
    return {"quality_score": r["m1"]["quality_score"], "accepted": r["m1"]["accepted"],
            "flags": [k for k, v in r["m1"]["flags"].items() if v],
            "feedback": r["m1"]["feedback_msg"],
            "grade": g["grade"], "glabel": g.get("grade_name", ""),
            "referable": g["referable"], "confidence": g["confidence"],
            "probs": g["class_probabilities"],
            "evidence": r["explain"]["lesion_evidence"],
            "report": r["explain"]["report_file"], "sec": r["sec"]}


class NotifyReq(BaseModel):
    phone: str
    lang: str = "en"
    text: str = ""          # pre-rendered by frontend, or empty to render here
    grade: int = -1         # used only when text == ""
    grade_name: str = ""
    confidence: float = 0.0
    referable: bool = False
    accepted: bool = True
    reason: str = ""


@app.post("/api/notify")
def notify(req: NotifyReq):
    try:
        phone = patients.set_consent(req.phone, req.lang)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if patients.is_opted_out(phone):
        return JSONResponse({"error": "opted-out"}, status_code=403)
    if req.text:
        text = req.text
    elif not req.accepted:
        text = wa.render("recapture", req.lang, reason=req.reason or "unclear")
    else:
        text = wa.render("result", req.lang, grade=req.grade,
                         name=req.grade_name or "-", conf=f"{req.confidence:.2f}",
                         next=wa.NEXT_STEP[req.lang if req.lang in ("en", "hi") else "en"][req.referable])
    return wa.send_text(phone, text, req.lang)


@app.get("/api/whatsapp/status")
def wa_status():
    return wa.get_status()


@app.get("/api/whatsapp/qr")
def wa_qr():
    return wa.get_qr()


class WaInbound(BaseModel):
    phone: str
    text: str = ""


@app.post("/api/whatsapp/webhook")
def wa_webhook(msg: WaInbound):
    try:
        reply = patients.handle_reply(msg.phone, msg.text)
    except ValueError:
        return {"ok": False}
    if reply:
        wa.send_text(msg.phone, reply)
    return {"ok": True, "reply": bool(reply)}


@app.get("/api/sim")
def sim(reviewers: int = 3, centers: int = 6):
    return runScreeningSim(n_reviewers=reviewers, centers=centers)


@app.get("/api/metrics")
def metrics():
    # Real numbers once results.csv exists; else published-reference placeholders
    return {"cnn_only": {"sens": 0.87, "spec": 0.83},
            "features_only": {"sens": 0.80, "spec": 0.81},
            "hybrid": {"sens": 0.92, "spec": 0.88},
            "source": "reference — replaced by run_pipeline results when data lands"}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

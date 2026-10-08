"""
app.py — Style Genie Flask application
"""
from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from PIL import Image

from utils.predict import FaceAnalysisError, analyze_face, _map_age_group
from utils.recommend import OCCASIONS, recommend_outfit

logging.basicConfig(level=logging.INFO)

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"]         = os.environ.get("STYLE_GENIE_SECRET", "style-genie-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

SKIN_TONES = [
    "Very Fair", "Fair", "Light", "Light Medium",
    "Medium", "Olive", "Tan", "Brown / Deep",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_image(file_data: bytes, suffix: str = ".jpg") -> Path:
    suffix = suffix.lower() if suffix else ".jpg"
    name   = f"{uuid.uuid4().hex}{suffix}"
    path   = UPLOAD_DIR / name
    img    = Image.open(io.BytesIO(file_data)).convert("RGB")
    img.thumbnail((1400, 1400), Image.LANCZOS)
    img.save(path, quality=92, optimize=True)
    return path


def _cleanup_old_uploads(keep: int = 50) -> None:
    files = sorted(UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try: old.unlink()
        except OSError: pass


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/upload")
def upload():
    return render_template("capture.html", mode="upload")

@app.get("/webcam")
def webcam():
    return render_template("capture.html", mode="webcam")

@app.get("/manual")
def manual():
    return render_template("manual.html", occasions=OCCASIONS, skin_tones=SKIN_TONES)

@app.get("/prediction")
def prediction():
    analysis = session.get("analysis")
    return render_template("prediction.html", analysis=analysis, occasions=OCCASIONS)

@app.get("/recommendation")
def recommendation():
    return render_template(
        "recommendation.html",
        recommendation=session.get("recommendation"),
        analysis=session.get("analysis"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API — Image analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
def api_analyze():
    try:
        if "image" in request.files:
            f = request.files["image"]
            if not f.filename:
                raise FaceAnalysisError("No file selected.")
            if f.mimetype not in ALLOWED_MIME:
                raise FaceAnalysisError("Only JPG, PNG, or WEBP images are accepted.")
            raw  = f.read()
            ext  = Path(f.filename).suffix.lower() or ".jpg"
            path = _save_image(raw, ext)
        else:
            payload  = request.get_json(silent=True) or {}
            data_url = payload.get("image", "")
            if not data_url.startswith("data:image/"):
                raise FaceAnalysisError("No image received.")
            header, encoded = data_url.split(",", 1)
            suffix = ".png" if "png" in header else ".jpg"
            path   = _save_image(base64.b64decode(encoded), suffix)

        _cleanup_old_uploads()
        analysis = analyze_face(path)
        analysis["image_url"] = f"/static/uploads/{path.name}"
        session["analysis"]   = analysis
        return jsonify({"ok": True, "analysis": analysis})

    except FaceAnalysisError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify({"ok": False, "error": "Something went wrong. Please try a clearer photo."}), 500


# ─────────────────────────────────────────────────────────────────────────────
# API — Edit detected attributes (from prediction page inline editors)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/edit-analysis")
def api_edit_analysis():
    """
    Allow user to correct AI-detected values.
    Accepts: { field: "age"|"gender"|"skin_tone", value: <string> }
    """
    analysis = session.get("analysis")
    if not analysis:
        return jsonify({"ok": False, "error": "No analysis in session."}), 400

    payload = request.get_json(silent=True) or {}
    field   = payload.get("field", "")
    value   = str(payload.get("value", "")).strip()

    if field == "age":
        try:
            age_int = int(float(value))
            if not (1 <= age_int <= 99):
                raise ValueError()
            analysis["age_group"]       = _map_age_group(age_int)
            analysis["confidence"]["age"] = 1.0   # user-provided = 100%
        except ValueError:
            return jsonify({"ok": False, "error": "Please enter a valid age (1–99)."}), 400

    elif field == "gender":
        if value not in ("Male", "Female"):
            return jsonify({"ok": False, "error": "Invalid gender value."}), 400
        analysis["gender"]               = value
        analysis["confidence"]["gender"] = 1.0

    elif field == "skin_tone":
        if value not in SKIN_TONES:
            return jsonify({"ok": False, "error": "Invalid skin tone."}), 400
        analysis["skin_tone"]                  = value
        analysis["confidence"]["skin_tone"]    = 1.0

    else:
        return jsonify({"ok": False, "error": "Unknown field."}), 400

    analysis["needs_input"] = (
        analysis.get("age_group") == "Needs input"
        or analysis.get("gender") == "Needs input"
    )
    session["analysis"] = analysis
    return jsonify({"ok": True, "analysis": analysis})


# ─────────────────────────────────────────────────────────────────────────────
# API — Manual entry (skip photo entirely)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/manual")
def api_manual():
    payload    = request.get_json(silent=True) or {}
    gender     = payload.get("gender", "").strip()
    skin_tone  = payload.get("skin_tone", "").strip()
    occasion   = payload.get("occasion", "").strip()

    try:
        age_int = int(float(payload.get("age", 0)))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "Please enter a valid age."}), 400

    errors = []
    if gender not in ("Male", "Female"):
        errors.append("Please select a gender.")
    if not (1 <= age_int <= 99):
        errors.append("Please enter an age between 1 and 99.")
    if skin_tone not in SKIN_TONES:
        errors.append("Please select a skin tone.")
    if occasion not in OCCASIONS:
        errors.append("Please select an occasion.")
    if errors:
        return jsonify({"ok": False, "error": " ".join(errors)}), 400

    analysis = {
        "age_group":  _map_age_group(age_int),
        "gender":     gender,
        "skin_tone":  skin_tone,
        "image_url":  None,
        "confidence": {"age": 1.0, "gender": 1.0, "skin_tone": 1.0},
        "face_box":   {},
        "model":      "Manual entry",
    }
    session["analysis"] = analysis

    recommendation = recommend_outfit(
        age_group = analysis["age_group"],
        gender    = gender,
        skin_tone = skin_tone,
        occasion  = occasion,
    )
    session["recommendation"] = recommendation
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
# API — Outfit recommendation
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/recommend")
def api_recommend():
    analysis = session.get("analysis")
    if not analysis:
        return jsonify({"ok": False, "error": "Please analyze a photo first."}), 400
    if analysis.get("needs_input"):
        return jsonify({"ok": False, "error": "Please enter your age and gender before continuing."}), 400

    payload  = request.get_json(silent=True) or {}
    occasion = payload.get("occasion", "").strip()

    if occasion not in OCCASIONS:
        return jsonify({"ok": False, "error": "Please select a valid occasion."}), 400

    recommendation = recommend_outfit(
        age_group = analysis["age_group"],
        gender    = analysis["gender"],
        skin_tone = analysis["skin_tone"],
        occasion  = occasion,
    )
    session["recommendation"] = recommendation
    return jsonify({"ok": True, "recommendation": recommendation})


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_): return jsonify({"ok": False, "error": "Image must be smaller than 10 MB."}), 413

@app.errorhandler(404)
def not_found(_): return render_template("index.html"), 404


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

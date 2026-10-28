"""Material Perception Experiment - Flask backend."""
from __future__ import annotations

import copy
import os
import random
import secrets
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory
from itsdangerous import BadSignature, URLSafeSerializer

from experiment_core import (
    BASE_DIR,
    BATCH_COUNT,
    BREAK_EVERY,
    CURRENT_STIMULUS_MANIFEST_ID,
    DEMO_MODE,
    STIM_CSV,
    TRIALS_PER_BATCH,
    assignment_debug_summary,
    build_observer_stimulus_label,
    build_response,
    create_session,
    current_trial,
    ensure_session_defaults,
    load_trials_from_csv,
    normalize_participant_id,
    ordered_results,
    resolve_image_path,
    session_uses_current_stimulus_manifest,
    utc_now_iso,
)
from storage import DemoStore, ResultStore
from stimulus_rendering import clamp_output_size, render_focus_encoded_bytes, resolve_repo_path

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
STORE = DemoStore() if DEMO_MODE else ResultStore()
MAX_DEMO_SESSIONS = 500
ADAPTATION_SECONDS = 20
SESSION_COOKIE_NAME = "material_perception_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
SESSION_COOKIE_SALT = "material-perception-session"
MAX_RECENT_ACTIONS = 40
STATIC_IMAGE_ROOT = BASE_DIR.resolve()
PRERENDERED_DIR = BASE_DIR / "prerendered"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
FOCUS_RENDER_VERSION = "20260424-hq1280"

GENDER_CODE = {"Male": "M", "Female": "F", "Non-binary": "N", "Prefer not to say": "P"}

# In-memory session cache keyed by participant_storage_id
_sessions: dict[str, dict] = {}
_session_serializer: URLSafeSerializer | None = None
_DEMO_SIGNING_SECRET = secrets.token_urlsafe(32)
TUTORIAL_CLASS_ORDER = (
    "Glass",
    "Polished Stone",
    "Metal",
)
TUTORIAL_RATINGS_BY_LABEL = {
    "Glass": {"gloss": 84, "trans": 72, "texture": 16},
    "Polished Stone": {"gloss": 76, "trans": 10, "texture": 44},
    "Metal": {"gloss": 82, "trans": 6, "texture": 28},
}


def generate_participant_id(gender: str, age: int) -> str:
    g = GENDER_CODE.get(gender, "X")
    a = str(int(age))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(100, 999)}"
    return f"{g}_{a}_{ts}_{suffix}"


_cached_stimulus_rows: list[dict] | None = None


def _read_stimulus_rows() -> list[dict]:
    global _cached_stimulus_rows
    if _cached_stimulus_rows is not None:
        return _cached_stimulus_rows

    if not STIM_CSV.exists():
        _cached_stimulus_rows = []
        return _cached_stimulus_rows

    import csv

    with STIM_CSV.open("r", newline="", encoding="utf-8") as handle:
        _cached_stimulus_rows = list(csv.DictReader(handle))
    return _cached_stimulus_rows


def _study_trial_count() -> int:
    rows = _read_stimulus_rows()
    if rows and {"batch_id", "is_anchor"}.issubset(rows[0].keys()):
        active = 0
        for row in rows:
            is_anchor = str(row.get("is_anchor", "")).strip().lower() in {"1", "true", "yes", "on"}
            batch_id = str(row.get("batch_id", "")).strip()
            if is_anchor or batch_id in {"1", "2", "3"}:
                active += 1
        return active
    return len(rows)


def _participant_trial_count() -> int:
    return min(TRIALS_PER_BATCH, _study_trial_count())


_prerendered_exists_cache: dict[str, bool] | None = None


def _scan_prerendered() -> dict[str, bool]:
    """Build a lookup of which pre-rendered files exist on disk."""
    global _prerendered_exists_cache
    if _prerendered_exists_cache is not None:
        return _prerendered_exists_cache
    _prerendered_exists_cache = {}
    if PRERENDERED_DIR.is_dir():
        for f in PRERENDERED_DIR.iterdir():
            if f.suffix == ".webp":
                _prerendered_exists_cache[f.name] = True
    return _prerendered_exists_cache


def _prerendered_url(stim_id: str, mode: str) -> str | None:
    """Return a static URL for a pre-rendered WebP stimulus."""
    if not stim_id:
        return None
    filename = f"{stim_id}_{mode}.webp"
    if _scan_prerendered().get(filename):
        return f"/api/prerendered/{filename}?v={FOCUS_RENDER_VERSION}"
    return None


def _render_focus_url(
    source_image_path: str,
    annotation_path: str,
    target_label: str,
    *,
    mode: str = "normal",
    stim_id: str = "",
) -> str:
    # Prefer pre-rendered static file
    if stim_id:
        static_url = _prerendered_url(stim_id, mode)
        if static_url:
            return static_url
    query = urlencode(
        {
            "src": source_image_path,
            "ann": annotation_path,
            "label": target_label,
            "mode": mode,
            "v": FOCUS_RENDER_VERSION,
        }
    )
    return f"/api/render-focus?{query}"


def _source_image_url(image_path: str) -> str:
    return f"/api/image/{image_path}"


def _dynamic_tutorial_trials() -> list[dict]:
    trials: list[dict] = []
    stimulus_rows = _read_stimulus_rows()
    used_trial_keys: set[str] = set()

    for class_label in TUTORIAL_CLASS_ORDER:
        sample = next(
            (
                row
                for row in stimulus_rows
                if row.get("class_label") == class_label
                and (
                    row.get("trial_key")
                    or f"{row.get('source_image_path', '')}|{row.get('annotation_path', '')}|{row.get('target_label', row.get('class_label', ''))}"
                )
                not in used_trial_keys
                and row.get("source_image_path")
                and row.get("annotation_path")
            ),
            None,
        )
        if sample is None:
            continue
        sample_key = sample.get("trial_key") or (
            f"{sample.get('source_image_path', '')}|"
            f"{sample.get('annotation_path', '')}|"
            f"{sample.get('target_label', sample.get('class_label', ''))}"
        )
        used_trial_keys.add(sample_key)
        _tsid = sample.get("stim_id", "")
        trials.append(
            {
                "classLabel": sample["class_label"],
                "imageUrl": _render_focus_url(
                    sample["source_image_path"],
                    sample["annotation_path"],
                    sample.get("target_label", sample["class_label"]),
                    mode="normal",
                    stim_id=_tsid,
                ),
                "blurUrl": _render_focus_url(
                    sample["source_image_path"],
                    sample["annotation_path"],
                    sample.get("target_label", sample["class_label"]),
                    mode="blur",
                    stim_id=_tsid,
                ),
                "ratings": TUTORIAL_RATINGS_BY_LABEL.get(
                    sample["class_label"],
                    {"gloss": 50, "trans": 50, "texture": 50},
                ),
            }
        )

    if trials:
        return trials

    # If the active dataset does not include the preferred tutorial classes,
    # use the first few real study rows so participants still see the same
    # focus/blur interaction they will rate.
    seen_classes: set[str] = set()
    for sample in stimulus_rows:
        class_label = sample.get("class_label", "Object") or "Object"
        if class_label in seen_classes:
            continue
        if not sample.get("source_image_path") or not sample.get("annotation_path"):
            continue
        seen_classes.add(class_label)
        _tsid = sample.get("stim_id", "")
        trials.append(
            {
                "classLabel": class_label,
                "imageUrl": _render_focus_url(
                    sample["source_image_path"],
                    sample["annotation_path"],
                    sample.get("target_label", class_label),
                    mode="normal",
                    stim_id=_tsid,
                ),
                "blurUrl": _render_focus_url(
                    sample["source_image_path"],
                    sample["annotation_path"],
                    sample.get("target_label", class_label),
                    mode="blur",
                    stim_id=_tsid,
                ),
                "ratings": TUTORIAL_RATINGS_BY_LABEL.get(
                    class_label,
                    {"gloss": 50, "trans": 50, "texture": 50},
                ),
            }
        )
        if len(trials) >= 3:
            return trials

    # Fallback for an empty or malformed manifest so the tutorial still renders.
    return [
        {
            "classLabel": "Object",
            "imageUrl": "",
            "blurUrl": "",
            "ratings": {"gloss": 50, "trans": 50, "texture": 50},
        }
    ]


def _path_exists(path_value: str) -> bool:
    try:
        return resolve_repo_path(path_value).exists()
    except (FileNotFoundError, ValueError):
        return False


def _remember_session(participant_storage_id: str, session: dict) -> None:
    _sessions[participant_storage_id] = session
    if DEMO_MODE:
        # Demo sessions exist only in memory; drop the oldest ones past the cap.
        while len(_sessions) > MAX_DEMO_SESSIONS:
            _sessions.pop(next(iter(_sessions)))


def _request_is_secure() -> bool:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    if forwarded_proto:
        return forwarded_proto.split(",")[0].strip().lower() == "https"
    return request.is_secure


def _session_signing_secret() -> str:
    secret = os.getenv("SESSION_SIGNING_SECRET", "").strip()
    if not secret:
        secret = os.getenv("SUPABASE_KEY", "").strip()
    if not secret and STORE.remote is not None:
        secret = str(getattr(STORE.remote, "key", "") or "").strip()
    if not secret and DEMO_MODE:
        # Demo sessions live only in memory, so a per-process key is enough.
        secret = _DEMO_SIGNING_SECRET
    if not secret:
        raise RuntimeError("SESSION_SIGNING_SECRET or SUPABASE_KEY must be configured.")
    return secret


def _cookie_serializer() -> URLSafeSerializer:
    global _session_serializer
    if _session_serializer is None:
        _session_serializer = URLSafeSerializer(_session_signing_secret(), salt=SESSION_COOKIE_SALT)
    return _session_serializer


def _sign_participant_storage_id(participant_storage_id: str) -> str:
    return _cookie_serializer().dumps(participant_storage_id)


def _decode_participant_storage_id(cookie_value: str) -> str:
    if not cookie_value:
        return ""
    try:
        return str(_cookie_serializer().loads(cookie_value)).strip()
    except BadSignature:
        return ""


def _set_session_cookie(response, participant_storage_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        _sign_participant_storage_id(participant_storage_id),
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=_request_is_secure(),
    )


def _clear_session_cookie(response) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        samesite="Lax",
        secure=_request_is_secure(),
    )


def _cookie_participant_storage_id() -> str:
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME, "").strip()
    return _decode_participant_storage_id(cookie_value)


def _missing_session_response(
    message: str = "Session not found.",
    status: int = 404,
    *,
    clear_cookie: bool = True,
):
    response = jsonify({"ok": False, "error": message})
    if clear_cookie:
        _clear_session_cookie(response)
    return response, status


def _resolve_static_image_path(img_path: str) -> Path:
    resolved = resolve_repo_path(img_path).resolve()
    resolved.relative_to(STATIC_IMAGE_ROOT)
    if resolved.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image type.")
    return resolved


def _cache_action_response(session: dict, request_id: str, response_payload: dict) -> None:
    recent_actions = session.setdefault("recent_actions", [])
    recent_actions = [item for item in recent_actions if item.get("request_id") != request_id]
    recent_actions.append(
        {
            "request_id": request_id,
            "response": copy.deepcopy(response_payload),
        }
    )
    session["recent_actions"] = recent_actions[-MAX_RECENT_ACTIONS:]


def _cached_action_response(session: dict, request_id: str) -> dict | None:
    if not request_id:
        return None
    for item in reversed(session.get("recent_actions", [])):
        if item.get("request_id") == request_id:
            return copy.deepcopy(item.get("response"))
    return None


def _load_session_by_id(participant_storage_id: str) -> tuple[dict | None, str | None]:
    if not participant_storage_id:
        return None, None

    cached = _sessions.get(participant_storage_id)
    if cached is not None:
        if not session_uses_current_stimulus_manifest(cached):
            _sessions.pop(participant_storage_id, None)
            return None, "Saved session uses an older stimulus set. Start a new session."
        return cached, "Resumed from memory."

    saved, load_msg = STORE.load_session(participant_storage_id)
    if saved is None:
        return None, load_msg

    session = ensure_session_defaults(saved, STORE.backend_label())
    if not session_uses_current_stimulus_manifest(session):
        _sessions.pop(participant_storage_id, None)
        return None, "Saved session uses an older stimulus set. Start a new session."
    session["participant_storage_id"] = normalize_participant_id(
        session.get("participant_storage_id") or session.get("participant_id", "")
    )
    session.setdefault("recent_actions", [])
    _sessions[participant_storage_id] = session
    return session, load_msg or "Resumed saved progress."


def _reload_session_from_store(participant_storage_id: str) -> tuple[dict | None, str | None]:
    if not participant_storage_id:
        return None, None

    saved, load_msg = STORE.load_session(participant_storage_id)
    if saved is None:
        return None, load_msg

    session = ensure_session_defaults(saved, STORE.backend_label())
    if not session_uses_current_stimulus_manifest(session):
        _sessions.pop(participant_storage_id, None)
        return None, "Saved session uses an older stimulus set. Start a new session."

    session["participant_storage_id"] = normalize_participant_id(
        session.get("participant_storage_id") or session.get("participant_id", "")
    )
    session.setdefault("recent_actions", [])
    _sessions[participant_storage_id] = session
    return session, load_msg or "Resumed saved progress."


def _request_session() -> tuple[dict | None, str | None, str]:
    participant_storage_id = _cookie_participant_storage_id()
    session, load_msg = _load_session_by_id(participant_storage_id)
    return session, load_msg, participant_storage_id


def _session_error_response(load_msg: str | None):
    if load_msg and "older stimulus set" in load_msg.lower():
        return _missing_session_response(load_msg, 404, clear_cookie=True)
    if load_msg:
        return _missing_session_response(load_msg, 503, clear_cookie=False)
    return _missing_session_response()


RESUME_TOKEN_SALT = "material-perception-resume"
_resume_serializer: URLSafeSerializer | None = None


def _resume_token_serializer() -> URLSafeSerializer:
    global _resume_serializer
    if _resume_serializer is None:
        _resume_serializer = URLSafeSerializer(
            _session_signing_secret(), salt=RESUME_TOKEN_SALT
        )
    return _resume_serializer


def _make_resume_token(participant_storage_id: str) -> str:
    return _resume_token_serializer().dumps(participant_storage_id)


def _decode_resume_token(token: str) -> str:
    try:
        return str(_resume_token_serializer().loads(token)).strip()
    except BadSignature:
        return ""


# ---------------------------------------------------------------------------
# Helper: build trial payload for the frontend
# ---------------------------------------------------------------------------

def _trial_payload(session: dict, status_message: str | None = None) -> dict:
    """Return JSON-safe dict describing the current trial state."""
    trial = current_trial(session)
    saved = session["responses"].get(trial["trial_key"], {})
    observer_stimulus_label = trial.get("observer_stimulus_label", "").strip() or build_observer_stimulus_label(
        trial.get("source_image_path", ""),
        trial.get("class_label", "Object"),
        trial.get("source_object_labels", ""),
        fallback_name=trial.get("source_sample_id", trial.get("stim_id", "stimulus")),
    )

    # Deterministic random starting positions for unseen trials (seeded per participant+trial)
    if saved:
        gv = float(saved.get("gloss_rating", 5))
        tv = float(saved.get("translucency_rating", 5))
        xv = float(saved.get("texture_rating", 5))
    else:
        seed_str = session.get("participant_id", "") + "|" + trial["trial_key"]
        rng = random.Random(seed_str)
        gv = rng.randint(0, 10)
        tv = rng.randint(0, 10)
        xv = rng.randint(0, 10)

    if trial.get("render_mode") == "labelme_focus":
        _sid = trial.get("stim_id", "")
        image_url = _render_focus_url(
            trial.get("source_image_path", ""),
            trial.get("annotation_path", ""),
            trial.get("target_label", trial["class_label"]),
            mode="normal",
            stim_id=_sid,
        )
        blur_url = _render_focus_url(
            trial.get("source_image_path", ""),
            trial.get("annotation_path", ""),
            trial.get("target_label", trial["class_label"]),
            mode="blur",
            stim_id=_sid,
        )
        image_missing = not (
            _prerendered_url(_sid, "normal") is not None
            or (
                _path_exists(trial.get("source_image_path", ""))
                and _path_exists(trial.get("annotation_path", ""))
            )
        )
    else:
        img_path = resolve_image_path(trial["image_path"])
        image_url = f"/api/image/{trial['image_path']}"
        blur_path = trial.get("blur_path", "")
        blur_url = f"/api/image/{blur_path}" if blur_path else ""
        image_missing = not Path(img_path).exists()

    is_last = session["current_index"] >= len(session["trials"]) - 1
    is_first = session["current_index"] == 0
    total = len(session["trials"])
    answered = len(ordered_results(session))
    cur = session["current_index"] + 1
    next_prefetch = None
    next_index = session["current_index"] + 1
    if 0 <= next_index < len(session["trials"]):
        next_trial = session["trials"][next_index]
        if next_trial.get("render_mode") == "labelme_focus":
            _nsid = next_trial.get("stim_id", "")
            next_prefetch = {
                "image_url": _render_focus_url(
                    next_trial.get("source_image_path", ""),
                    next_trial.get("annotation_path", ""),
                    next_trial.get("target_label", next_trial["class_label"]),
                    mode="normal",
                    stim_id=_nsid,
                ),
                "blur_url": _render_focus_url(
                    next_trial.get("source_image_path", ""),
                    next_trial.get("annotation_path", ""),
                    next_trial.get("target_label", next_trial["class_label"]),
                    mode="blur",
                    stim_id=_nsid,
                ),
            }
        else:
            next_blur_path = next_trial.get("blur_path", "")
            next_prefetch = {
                "image_url": f"/api/image/{next_trial['image_path']}",
                "blur_url": f"/api/image/{next_blur_path}" if next_blur_path else "",
            }

    # Break message
    break_msg = ""
    idx = session["current_index"]
    if idx > 0 and idx % BREAK_EVERY == 0:
        break_msg = "Take a short break if needed. Your progress is already saved."
    if total and idx == total - 1:
        if break_msg:
            break_msg += " "
        break_msg += "This is the final trial. Press Submit when you are done."

    return {
        "trial_number": cur,
        "total_trials": total,
        "batch_number": session.get("batch_number"),
        "batch_count": session.get("batch_count", BATCH_COUNT),
        "answered": answered,
        "progress_pct": round(cur / total * 100, 1) if total else 0,
        "trial_key": trial["trial_key"],
        "class_label": trial["class_label"],
        "target_label": trial.get("target_label", trial["class_label"]),
        "stimulus_display_name": trial.get("stimulus_display_name", trial.get("stimulus_name", trial["stim_id"])),
        "observer_stimulus_label": observer_stimulus_label,
        "image_url": image_url,
        "source_image_url": _source_image_url(trial.get("source_image_path", "")) if trial.get("render_mode") == "labelme_focus" else image_url,
        "blur_url": blur_url,
        "prefetch_next": next_prefetch,
        "image_missing": image_missing,
        "gloss_value": gv,
        "translucency_value": tv,
        "texture_value": xv,
        "is_first": is_first,
        "is_last": is_last,
        "break_message": break_msg,
        "status_message": status_message or "",
        "completed": False,
    }


def _completion_payload(session: dict, save_msg: str = "") -> dict:
    results = ordered_results(session)
    at = session.get("completed_at") or utc_now_iso()
    return {
        "completed": True,
        "trials_completed": len(results),
        "total_trials": len(session["trials"]),
        "batch_number": session.get("batch_number"),
        "batch_count": session.get("batch_count", BATCH_COUNT),
        "completed_at": at,
        "storage_backend": session.get("storage_backend", ""),
        "save_message": save_msg,
        "experiment_comment": session.get("experiment_comment", ""),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        adaptation_seconds=ADAPTATION_SECONDS,
        study_trial_count=_study_trial_count(),
        participant_trial_count=_participant_trial_count(),
        batch_count=BATCH_COUNT,
        demo_mode=DEMO_MODE,
        tutorial_trials=_dynamic_tutorial_trials(),
    )


@app.route("/resume/<token>")
def resume_via_link(token):
    """Set the session cookie from a signed resume token and redirect to /."""
    participant_storage_id = _decode_resume_token(token)
    if not participant_storage_id:
        return "Invalid or expired resume link.", 403

    # Verify session actually exists in storage
    session, _ = _load_session_by_id(participant_storage_id)
    if session is None:
        return "Session not found.", 404

    response = redirect("/")
    _set_session_cookie(response, participant_storage_id)
    return response


@app.route("/api/resume-token")
def get_resume_token():
    """Return the resume URL for the current session."""
    session, _, participant_storage_id = _request_session()
    if not session:
        return jsonify({"ok": False, "error": "No active session."}), 404
    token = _make_resume_token(participant_storage_id)
    return jsonify({"ok": True, "resume_token": token})


@app.route("/api/image/<path:img_path>")
def serve_image(img_path):
    """Serve stimulus images from inside the repository."""
    try:
        p = _resolve_static_image_path(img_path)
    except ValueError:
        return "Forbidden", 403
    if not p.exists():
        return "Not found", 404
    return send_from_directory(str(p.parent), p.name, max_age=86400)


@app.route("/api/prerendered/<filename>")
def serve_prerendered(filename):
    """Serve pre-rendered stimulus WebP files with long cache."""
    if not filename.endswith(".webp") or "/" in filename or "\\" in filename:
        return "Forbidden", 403
    p = PRERENDERED_DIR / filename
    if not p.exists():
        return "Not found", 404
    response = send_from_directory(str(PRERENDERED_DIR), filename, mimetype="image/webp", max_age=31536000)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.route("/api/render-focus")
def render_focus():
    source_image_path = request.args.get("src", "").strip()
    annotation_path = request.args.get("ann", "").strip()
    target_label = request.args.get("label", "").strip()
    mode = request.args.get("mode", "normal").strip().lower()
    fmt = request.args.get("fmt", "png").strip().lower()
    quality_raw = request.args.get("q", "90").strip()
    width_raw = request.args.get("w", "").strip()
    height_raw = request.args.get("h", "").strip()

    if not source_image_path or not annotation_path or not target_label:
        return jsonify({"ok": False, "error": "Missing render parameters."}), 400
    if mode not in {"normal", "blur"}:
        return jsonify({"ok": False, "error": "Invalid render mode."}), 400
    if fmt not in {"png", "jpeg", "jpg", "webp"}:
        return jsonify({"ok": False, "error": "Invalid render format."}), 400

    try:
        output_width = int(width_raw) if width_raw else None
        output_height = int(height_raw) if height_raw else None
        output_size = clamp_output_size(output_width, output_height)
        quality = int(quality_raw or "90")
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid render size or quality."}), 400

    try:
        image_format = "JPEG" if fmt in {"jpeg", "jpg"} else fmt.upper()
        image_bytes = render_focus_encoded_bytes(
            source_image_path,
            annotation_path,
            target_label,
            mode,
            output_size[0],
            output_size[1],
            image_format,
            quality,
        )
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Source image not found."}), 404
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    mime = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }.get(image_format, "image/png")
    response = send_file(BytesIO(image_bytes), mimetype=mime, max_age=0, conditional=False)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/session", methods=["POST"])
def create_session_route():
    if not STORE.is_configured():
        return jsonify(
            {
                "ok": False,
                "errors": ["Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY before collecting data."],
            }
        ), 503

    data = request.get_json(force=True)

    device = data.get("device")
    age = data.get("age")
    gender = data.get("gender")
    cvd = data.get("cvd", "No")
    cvd_type = data.get("cvd_type")
    glasses = data.get("glasses", "No correction needed")
    color_scientist = data.get("color_scientist")
    consent = data.get("consent", False)

    # Validation
    errors = []
    if not consent:
        errors.append("You must consent before proceeding.")
    if not device:
        errors.append("Select your device type.")
    try:
        age = int(age) if age is not None else None
    except (ValueError, TypeError):
        age = None
    if age is None or age < 18 or age > 100:
        errors.append("Enter a valid age (18-100).")
    if not gender:
        errors.append("Select your gender.")
    if not color_scientist:
        errors.append("Answer whether you are a color scientist.")
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    if DEMO_MODE:
        # The demo shows every bundled stimulus once, in a shuffled order.
        batch_number = None
        allocator_message = "demo"
    else:
        allocation = STORE.allocate_batch(BATCH_COUNT, CURRENT_STIMULUS_MANIFEST_ID)
        if not allocation["ok"]:
            return jsonify({"ok": False, "errors": [allocation["message"]]}), 503
        batch_number = int(allocation["batch_number"])
        allocator_message = allocation.get("message", "")

    pid = generate_participant_id(gender, age)
    session = create_session(pid, storage_backend=STORE.backend_label(), batch_number=batch_number)
    participant_storage_id = normalize_participant_id(pid)
    session["participant_storage_id"] = participant_storage_id
    session["recent_actions"] = []
    session["batch_allocator"] = allocator_message

    session["demographics"] = {
        "device": device,
        "age": age,
        "gender": gender,
        "color_vision_deficiency": cvd,
        "cvd_type": cvd_type if cvd == "Yes" else None,
        "glasses": glasses,
        "color_scientist": color_scientist,
    }

    save_result = STORE.save_session(session)
    if not save_result["ok"]:
        return jsonify({"ok": False, "errors": [save_result["message"]]}), 503

    _remember_session(participant_storage_id, session)

    response = jsonify({
        "ok": True,
        "resume_message": "Session created.",
        "trial": _trial_payload(session, save_result["message"]),
        "resume_token": _make_resume_token(participant_storage_id),
    })
    _set_session_cookie(response, participant_storage_id)
    return response


@app.route("/api/session", methods=["GET"])
def get_current_session():
    participant_storage_id = _cookie_participant_storage_id()
    if not participant_storage_id:
        return _missing_session_response()
    session, load_msg = _load_session_by_id(participant_storage_id)
    if not session:
        return _session_error_response(load_msg)
    if session.get("completed"):
        return jsonify({"ok": True, **_completion_payload(session)})
    return jsonify({
        "ok": True,
        "trial": _trial_payload(session, load_msg),
    })


@app.route("/api/session", methods=["DELETE"])
def clear_current_session():
    participant_storage_id = _cookie_participant_storage_id()
    if participant_storage_id:
        _sessions.pop(participant_storage_id, None)
    response = jsonify({"ok": True})
    _clear_session_cookie(response)
    return response


@app.route("/api/trial")
@app.route("/api/trial/<session_id>")
def get_trial(session_id=None):
    session, load_msg, _ = _request_session()
    if not session:
        return _session_error_response(load_msg)
    if session.get("completed"):
        return jsonify({"ok": True, **_completion_payload(session)})
    return jsonify({"ok": True, **_trial_payload(session, load_msg)})


@app.route("/api/response", methods=["POST"])
def save_response():
    return _handle_response(request.get_json(force=True))


def _handle_response(data):
    """Core logic for saving a response and advancing/retreating the trial."""
    session, load_msg, participant_storage_id = _request_session()
    if not session:
        return _session_error_response(load_msg)

    try:
        g = round(max(0.0, min(10.0, float(data.get("gloss", 5)))), 2)
        t = round(max(0.0, min(10.0, float(data.get("translucency", 5)))), 2)
        x = round(max(0.0, min(10.0, float(data.get("texture", 5)))), 2)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "Invalid rating values."}), 400

    request_id = str(data.get("request_id", "")).strip()
    expected_trial_key = str(data.get("trial_key", "")).strip()
    if not request_id:
        return jsonify({"ok": False, "error": "Missing request_id."}), 400
    if not expected_trial_key:
        return jsonify({"ok": False, "error": "Missing trial_key."}), 400

    cached_response = _cached_action_response(session, request_id)
    if cached_response is not None:
        return jsonify(cached_response)

    active_trial_key = current_trial(session)["trial_key"]
    if expected_trial_key != active_trial_key:
        refreshed_session, refreshed_msg = _reload_session_from_store(participant_storage_id)
        if refreshed_session is not None:
            session = refreshed_session
            load_msg = refreshed_msg
            cached_response = _cached_action_response(session, request_id)
            if cached_response is not None:
                return jsonify(cached_response)
            active_trial_key = current_trial(session)["trial_key"]
        if expected_trial_key != active_trial_key:
            return jsonify({"ok": False, "error": "This trial changed before your rating was saved. Reload the current trial and try again."}), 409

    working_session = copy.deepcopy(session)

    rt = time.time() - session.get("trial_started_at", time.time())
    resp = build_response(
        working_session,
        gloss_rating=g,
        translucency_rating=t,
        texture_rating=x,
        rt_sec=rt,
    )
    working_session["responses"][resp["trial_key"]] = resp

    action = data.get("action", "next")  # "next", "previous", "submit"

    if action == "submit":
        working_session["experiment_comment"] = str(data.get("experiment_comment", "") or "").strip()[:200]
        working_session["completed"] = True
        working_session["completed_at"] = utc_now_iso()
        working_session["updated_at"] = utc_now_iso()
    elif action == "previous":
        working_session["current_index"] = max(0, working_session["current_index"] - 1)
    else:
        working_session["current_index"] = min(
            len(working_session["trials"]) - 1,
            working_session["current_index"] + 1,
        )
        action = "next"

    working_session["trial_started_at"] = time.time()
    working_session["updated_at"] = utc_now_iso()

    success_message = "Demo mode: responses are not saved." if DEMO_MODE else "Saved to Supabase."
    if action == "submit":
        response_payload = {"ok": True, **_completion_payload(working_session, success_message)}
    else:
        response_payload = {"ok": True, **_trial_payload(working_session, success_message)}
    _cache_action_response(working_session, request_id, response_payload)

    save_result = STORE.save_session(working_session)
    if not save_result["ok"]:
        return jsonify({"ok": False, "error": save_result["message"]}), 503

    if action == "submit":
        _sessions.pop(participant_storage_id, None)
        response = jsonify(response_payload)
        _clear_session_cookie(response)
        return response

    _remember_session(participant_storage_id, working_session)
    return jsonify(response_payload)


@app.route("/api/previous", methods=["POST"])
def go_previous():
    data = request.get_json(force=True)
    data["action"] = "previous"
    return _handle_response(data)


@app.route("/api/resume", methods=["POST"])
def resume_session():
    session, load_msg, _ = _request_session()
    if not session:
        return _session_error_response(load_msg)
    if session.get("completed"):
        return jsonify({"ok": True, **_completion_payload(session)})
    return jsonify({
        "ok": True,
        "trial": _trial_payload(session, load_msg),
    })


@app.route("/api/status")
@app.route("/api/status/<session_id>")
def get_status(session_id=None):
    session, load_msg, _ = _request_session()
    if not session:
        return _session_error_response(load_msg)
    return jsonify({
        "ok": True,
        "completed": session.get("completed", False),
        "current_index": session["current_index"],
        "total_trials": len(session["trials"]),
        "answered": len(ordered_results(session)),
    })


def _assignment_debug_allowed() -> bool:
    if os.environ.get("ASSIGNMENT_DEBUG_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if app.debug:
        return True
    return request.remote_addr in {"127.0.0.1", "::1", "localhost"}


@app.route("/api/debug/assignment")
def debug_assignment():
    """Inspect deterministic batch assignment locally or when explicitly enabled."""
    if not _assignment_debug_allowed():
        return jsonify({"ok": False, "error": "Assignment debug endpoint is disabled."}), 403

    raw_batch = request.args.get("batch", "1").strip()
    participant_id = request.args.get("participant_id", "debug_assignment").strip() or "debug_assignment"
    try:
        batch_number = int(raw_batch)
        trials = load_trials_from_csv(
            STIM_CSV,
            participant_id=participant_id,
            batch_number=batch_number,
        )
    except (TypeError, ValueError, RuntimeError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    summary = assignment_debug_summary(trials)
    return jsonify(
        {
            "ok": summary["valid"],
            "participant_id": participant_id,
            "batch_number": batch_number,
            "stimulus_manifest_id": CURRENT_STIMULUS_MANIFEST_ID,
            "summary": summary,
            "trials": [
                {
                    "trial_index": idx,
                    "presented_order_index": idx + 1,
                    "trial_key": trial.get("trial_key"),
                    "stimulus_id": trial.get("stimulus_id", trial.get("stim_id")),
                    "material_class": trial.get("material_class", trial.get("class_label")),
                    "is_anchor": bool(trial.get("is_anchor")),
                    "is_qc_repeat": bool(trial.get("is_qc_repeat")),
                    "assignment_role": trial.get("assignment_role"),
                    "repeated_from_trial_key": trial.get("repeated_from_trial_key"),
                    "repeated_from_trial_index": trial.get("repeated_from_trial_index"),
                    "qc_gap": trial.get("qc_gap"),
                }
                for idx, trial in enumerate(trials)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=os.getenv("FLASK_DEBUG") == "1")

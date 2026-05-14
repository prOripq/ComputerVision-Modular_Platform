import json
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import cv2
import requests
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)

from auth import create_default_admin, verify_password
from core.database import DatabaseManager
from core.video_buffer import VideoBuffer
from core.video_loader import VideoLoader
from modules.face_module import FaceRecognizer
from modules.person_module import PersonDetector
from modules.zone_module import ZoneDetector, TripLine, Zone, load_zones, save_zones

# ---------------------------------------------------------------------------
# Логгер
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask — SECRET_KEY сохраняется в файл, чтобы сессии не слетали при рестарте
# ---------------------------------------------------------------------------
app = Flask(__name__)

_SECRET_FILE = ".flask_secret"
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    if os.path.exists(_SECRET_FILE):
        with open(_SECRET_FILE, "r") as _f:
            _secret = _f.read().strip()
    else:
        _secret = secrets.token_hex(32)
        with open(_SECRET_FILE, "w") as _f:
            _f.write(_secret)
        logger.warning("Новый SECRET_KEY сохранён в %s", _SECRET_FILE)

app.secret_key = _secret
app.permanent_session_lifetime = 86400 * 7  # 7 дней

CONFIG_FILE = "config.json"
KNOWN_FACES_DIR = "known_faces"
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}

DEFAULT_CONFIG: dict = {
    "camera_source":       "0",
    "enable_face_rec":     True,
    "enable_people_count": True,
    "face_cooldown":       15.0,
    "log_interval":        5.0,
    "face_threshold":      0.55,
    "telegram_token":      "",
    "telegram_chat_id":    "",
    "enable_telegram":     False,
}

# ---------------------------------------------------------------------------
# Глобальное состояние
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
video_buffer:    VideoBuffer    | None = None
camera:          VideoLoader    | None = None
person_tracker:  PersonDetector | None = None
face_recognizer: FaceRecognizer | None = None
db:              DatabaseManager | None = None
zone_detector:   ZoneDetector   | None = None
config:          dict = {}
current_people_count: int = 0

_tg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tg_alert")

# Brute-force protection: {ip: [timestamps]}
_login_attempts: dict[str, list] = {}
_MAX_LOGIN_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300  # 5 минут


# ---------------------------------------------------------------------------
# Декоратор — защита роутов
# ---------------------------------------------------------------------------

def login_required(f):
    """Перенаправляет на /login если пользователь не авторизован."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

def load_config() -> dict:
    global config
    if not os.path.exists(CONFIG_FILE):
        config = DEFAULT_CONFIG.copy()
        _write_config(config)
    else:
        with open(CONFIG_FILE, "r") as f:
            loaded = json.load(f)
        config = {**DEFAULT_CONFIG, **loaded}
    return config


def _write_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)


def save_config(new_config: dict) -> None:
    global config
    _write_config(new_config)
    config = new_config


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def init_system() -> None:
    global camera, person_tracker, face_recognizer, db, video_buffer, zone_detector

    create_default_admin()
    load_config()
    logger.info("--- ЗАГРУЗКА СИСТЕМЫ ---")

    src = config["camera_source"]
    src = int(src) if str(src).isdigit() else src

    try:
        camera = VideoLoader(src)
    except Exception:
        logger.exception("Не удалось открыть источник видео: %s", src)

    try:
        db = DatabaseManager()
    except Exception:
        logger.exception("Не удалось подключиться к базе данных.")

    try:
        person_tracker = PersonDetector()
    except Exception:
        logger.exception("Не удалось загрузить PersonDetector.")

    try:
        threshold = config.get("face_threshold", 0.55)
        face_recognizer = FaceRecognizer(db_path=KNOWN_FACES_DIR, similarity_threshold=threshold)
    except Exception:
        logger.exception("Не удалось загрузить FaceRecognizer.")

    try:
        zone_detector = ZoneDetector()
    except Exception:
        logger.exception("Не удалось загрузить ZoneDetector.")

    video_buffer = VideoBuffer(
        output_dir   = "clips",
        pre_seconds  = 10.0,
        post_seconds = 10.0,
        fps          = 25.0,
        cooldown     = 15.0,
    )

    logger.info("Система готова.")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _send_telegram_alert(token: str, chat_id: str, message: str, image_frame) -> None:
    try:
        ret, buffer = cv2.imencode(".jpg", image_frame)
        if not ret:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            files={"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")},
            data={"chat_id": chat_id, "caption": message},
            timeout=10,
        )
    except Exception:
        logger.exception("Ошибка при отправке Telegram-уведомления.")


# ---------------------------------------------------------------------------
# Генератор кадров
# ---------------------------------------------------------------------------

def generate_frames():
    global current_people_count

    last_log_people_time = time.time()
    last_seen_faces: dict[str, float] = {}
    last_cleanup_time = time.time()
    CLEANUP_INTERVAL = 300.0

    while True:
        if camera is None:
            time.sleep(0.1)
            continue

        frame = camera.get_frame()
        if frame is None:
            break

        cfg = config

        if cfg.get("enable_people_count") and person_tracker is not None:
            track_ids, boxes_xywh, raw_plot = person_tracker.detect(frame)

            if zone_detector and (zone_detector.zones or zone_detector.lines):
                frame = raw_plot
                frame, zone_stats = zone_detector.process(frame, track_ids, boxes_xywh)
                people_count = zone_stats["total"] if zone_stats["zones"] else len(track_ids)

                if (time.time() - last_log_people_time) > float(cfg["log_interval"]):
                    if db:
                        for zs in zone_stats["zones"]:
                            db.log_event("ZONE_STATS", f"{zs['name']}:{zs['count']}")
                        for ls in zone_stats["lines"]:
                            db.log_event("LINE_CROSS", f"{ls['name']}:in={ls['in']},out={ls['out']}")
                    last_log_people_time = time.time()
            else:
                frame = person_tracker.draw_tails(raw_plot, track_ids, boxes_xywh)
                people_count = len(track_ids)

                if (time.time() - last_log_people_time) > float(cfg["log_interval"]):
                    if db:
                        db.log_event("PEOPLE_STATS", str(people_count))
                    last_log_people_time = time.time()

            with _state_lock:
                current_people_count = people_count

        if cfg.get("enable_face_rec") and face_recognizer is not None:
            frame, found_names = face_recognizer.recognize(frame)
            curr_time = time.time()

            for name in found_names:
                if (curr_time - last_seen_faces.get(name, 0.0)) > float(cfg["face_cooldown"]):
                    if db:
                        db.log_event("FACE_MATCH", name)
                    logger.info("Опознан: %s", name)
                    last_seen_faces[name] = curr_time

                    if video_buffer is not None:
                        video_buffer.trigger(label=f"face_{name}")

                    if cfg.get("enable_telegram") and cfg.get("telegram_token"):
                        _tg_executor.submit(
                            _send_telegram_alert,
                            cfg["telegram_token"],
                            cfg["telegram_chat_id"],
                            f"🚨 ВНИМАНИЕ! Обнаружен: {name}",
                            frame.copy(),
                        )

        if video_buffer is not None:
            video_buffer.push(frame)

        if (time.time() - last_cleanup_time) > CLEANUP_INTERVAL:
            cutoff = time.time() - float(cfg["face_cooldown"]) * 10
            last_seen_faces = {k: v for k, v in last_seen_faces.items() if v > cutoff}
            last_cleanup_time = time.time()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Роуты — Аутентификация
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authenticated"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        now = time.time()

        # Очищаем старые попытки
        _login_attempts.setdefault(ip, [])
        _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < _LOCKOUT_SECONDS]

        if len(_login_attempts[ip]) >= _MAX_LOGIN_ATTEMPTS:
            remaining = int(_LOCKOUT_SECONDS - (now - _login_attempts[ip][0]))
            logger.warning("Блокировка входа для IP %s", ip)
            error = f"Слишком много попыток. Подождите {remaining} сек."
            return render_template("login.html", error=error)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if verify_password(username, password):
            _login_attempts.pop(ip, None)
            session.permanent = True
            session["authenticated"] = True
            session["username"] = username
            logger.info("Успешный вход: %s", username)
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        else:
            _login_attempts[ip].append(now)
            remaining_attempts = _MAX_LOGIN_ATTEMPTS - len(_login_attempts[ip])
            logger.warning("Неудачная попытка входа: '%s' (IP: %s)", username, ip)
            error = f"Неверный логин или пароль. Осталось попыток: {max(remaining_attempts, 0)}"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    session.clear()
    logger.info("Выход: %s", username)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Роуты — Основные страницы
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
@login_required
def api_stats():
    with _state_lock:
        people = current_people_count

    if db is None:
        return jsonify({"error": "db_unavailable"}), 503

    try:
        events_raw = db.get_recent_events(limit=15)
        last_face = db.get_last_face_match() or "Нет данных"
    except Exception:
        logger.exception("Ошибка при получении статистики из БД.")
        return jsonify({"error": "db_error"}), 500

    events = [
        {
            "id":      row["id"],
            "time":    row["timestamp"].split(" ")[1] if " " in row["timestamp"] else row["timestamp"],
            "type":    row["event_type"],
            "details": row["details"],
        }
        for row in events_raw
    ]

    return jsonify({
        "people_count": people,
        "last_face":    last_face,
        "events":       events,
    })


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    global camera

    if request.method == "POST":
        old_source = config["camera_source"]
        new_source = request.form.get("camera_source", old_source)

        new_config = config.copy()
        new_config["camera_source"]       = new_source
        new_config["enable_face_rec"]     = "enable_face_rec" in request.form
        new_config["enable_people_count"] = "enable_people_count" in request.form
        new_config["enable_telegram"]     = "enable_telegram" in request.form
        new_config["telegram_token"]      = request.form.get("telegram_token", "").strip()
        new_config["telegram_chat_id"]    = request.form.get("telegram_chat_id", "").strip()
        try:
            new_config["face_cooldown"]  = max(1.0, float(request.form.get("face_cooldown", 15.0)))
            new_config["log_interval"]   = max(1.0, float(request.form.get("log_interval", 5.0)))
            new_config["face_threshold"] = max(0.1, min(1.0, float(request.form.get("face_threshold", 0.55))))
        except ValueError:
            pass

        save_config(new_config)

        # Применяем порог распознавания сразу без перезапуска
        if face_recognizer is not None:
            face_recognizer.similarity_threshold = new_config["face_threshold"]

        if new_source != old_source:
            logger.info("Смена камеры: %s → %s", old_source, new_source)
            if camera:
                camera.release()
            src = int(new_source) if str(new_source).isdigit() else new_source
            camera = VideoLoader(src)

        return redirect(url_for("index"))

    return render_template("settings.html", config=config)


@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics")
@login_required
def api_analytics():
    if db is None:
        return jsonify({"error": "db_unavailable"}), 503
    try:
        period = int(request.args.get("days", 7))
        period = max(1, min(period, 90))
        return jsonify({
            "summary":   db.get_summary_stats(),
            "hourly":    db.get_hourly_stats(days=period),
            "daily":     db.get_daily_stats(days=period),
            "top_faces": db.get_top_faces(limit=10),
        })
    except Exception:
        logger.exception("Ошибка при получении аналитики.")
        return jsonify({"error": "db_error"}), 500


# ---------------------------------------------------------------------------
# Роуты — Зоны
# ---------------------------------------------------------------------------

@app.route("/zones")
@login_required
def zones_editor():
    return render_template("zones_editor.html")


@app.route("/api/zones", methods=["GET"])
@login_required
def api_zones_get():
    zones_list, lines_list = load_zones()
    return jsonify({
        "zones": [z.__dict__ for z in zones_list],
        "lines": [l.__dict__ for l in lines_list],
    })


@app.route("/api/zones", methods=["POST"])
@login_required
def api_zones_post():
    try:
        data = request.get_json()
        zones_list = [Zone(**z) for z in data.get("zones", [])]
        lines_list = [TripLine(**l) for l in data.get("lines", [])]
        save_zones(zones_list, lines_list)

        if zone_detector:
            zone_detector.reload()

        logger.info("Зоны обновлены: %d зон, %d линий", len(zones_list), len(lines_list))
        return jsonify({"status": "ok"})
    except Exception:
        logger.exception("Ошибка сохранения зон.")
        return jsonify({"error": "save_failed"}), 500


@app.route("/api/zones/reset_counters", methods=["POST"])
@login_required
def api_zones_reset():
    if zone_detector:
        zone_detector.reset_counters()
    return jsonify({"status": "ok"})


@app.route("/snapshot")
@login_required
def snapshot():
    if camera is None:
        return "", 503
    frame = camera.get_frame()
    if frame is None:
        return "", 503
    ret, buf = cv2.imencode(".jpg", frame)
    if not ret:
        return "", 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Роуты — Управление лицами
# ---------------------------------------------------------------------------

@app.route("/faces")
@login_required
def faces_page():
    return render_template("faces.html")


@app.route("/api/faces", methods=["GET"])
@login_required
def api_faces_list():
    """Список известных лиц."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    faces = []
    for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ALLOWED_IMAGE_EXT:
            name = os.path.splitext(filename)[0]
            faces.append({"name": name, "filename": filename})
    return jsonify(faces)


@app.route("/api/faces/upload", methods=["POST"])
@login_required
def api_faces_upload():
    """Загрузить фото нового человека."""
    if "photo" not in request.files:
        return jsonify({"error": "no_file"}), 400

    file = request.files["photo"]
    name = request.form.get("name", "").strip()

    if not name:
        return jsonify({"error": "no_name"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "invalid_format"}), 400

    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        return jsonify({"error": "invalid_name"}), 400

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    filepath = os.path.join(KNOWN_FACES_DIR, safe_name + ext)
    file.save(filepath)
    logger.info("Загружено фото для '%s'", safe_name)

    if face_recognizer is not None:
        face_recognizer.refresh_database()

    return jsonify({"status": "ok", "name": safe_name})


@app.route("/api/faces/<filename>", methods=["DELETE"])
@login_required
def api_faces_delete(filename):
    """Удалить лицо из базы."""
    safe_filename = os.path.basename(filename)
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "invalid_file"}), 400

    filepath = os.path.join(KNOWN_FACES_DIR, safe_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "not_found"}), 404

    os.remove(filepath)
    logger.info("Удалено лицо: %s", safe_filename)

    if face_recognizer is not None:
        face_recognizer.refresh_database()

    return jsonify({"status": "ok"})


@app.route("/api/faces/photo/<filename>")
@login_required
def api_faces_photo(filename):
    """Фото лица для предпросмотра."""
    return send_from_directory(os.path.abspath(KNOWN_FACES_DIR), os.path.basename(filename))


@app.route("/api/refresh_faces", methods=["POST"])
@login_required
def refresh_faces():
    try:
        if face_recognizer is None:
            return jsonify({"error": "not_loaded"}), 503
        face_recognizer.refresh_database()
        return jsonify({"status": "ok", "count": len(face_recognizer._names)})
    except Exception:
        logger.exception("Ошибка при обновлении базы лиц.")
        return jsonify({"error": "refresh_failed"}), 500


# ---------------------------------------------------------------------------
# Роуты — Клипы
# ---------------------------------------------------------------------------

@app.route("/clips_viewer")
@login_required
def clips_viewer():
    return render_template("clips.html")


@app.route("/clips")
@login_required
def list_clips():
    clips_dir = "clips"
    if not os.path.exists(clips_dir):
        return jsonify([])
    result = []
    for f in sorted(os.listdir(clips_dir), reverse=True):
        if not f.endswith(".mp4"):
            continue
        path = os.path.join(clips_dir, f)
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
        mtime = os.path.getmtime(path)
        result.append({
            "filename": f,
            "size_mb":  size_mb,
            "created":  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        })
    return jsonify(result)


@app.route("/clips/<path:filename>")
@login_required
def download_clip(filename):
    return send_from_directory(
        os.path.abspath("clips"),
        filename,
        as_attachment=False,
    )


# ---------------------------------------------------------------------------
# Смена пароля
# ---------------------------------------------------------------------------

@app.route("/api/change_password", methods=["POST"])
@login_required
def api_change_password():
    from auth import change_password as auth_change_password
    data = request.get_json() or {}
    old_pw = data.get("old_password", "")
    new_pw = data.get("new_password", "")
    if len(new_pw) < 6:
        return jsonify({"error": "too_short"}), 400
    username = session.get("username", "")
    if auth_change_password(username, old_pw, new_pw):
        logger.info("Пароль изменён для пользователя '%s'", username)
        return jsonify({"status": "ok"})
    return jsonify({"error": "wrong_password"}), 403


# ---------------------------------------------------------------------------
# Экспорт аналитики
# ---------------------------------------------------------------------------

@app.route("/api/analytics/export")
@login_required
def export_analytics():
    import csv, io
    if db is None:
        return jsonify({"error": "db_unavailable"}), 503
    try:
        period = int(request.args.get("days", 30))
        period = max(1, min(period, 90))
        daily = db.get_daily_stats(days=period)
        top   = db.get_top_faces(limit=50)

        output = io.StringIO()
        output.write(f"# CV Platform Analytics Export — last {period} days\n")
        output.write("\nDAILY STATS\n")
        writer = csv.DictWriter(output, fieldnames=["date", "max_people", "face_events"])
        writer.writeheader()
        writer.writerows(daily)
        output.write("\nTOP FACES\n")
        writer2 = csv.DictWriter(output, fieldnames=["name", "count", "last_seen"])
        writer2.writeheader()
        writer2.writerows(top)

        resp = Response(output.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = f"attachment; filename=analytics_{period}d.csv"
        return resp
    except Exception:
        logger.exception("Ошибка экспорта аналитики.")
        return jsonify({"error": "export_failed"}), 500


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_system()
    app.run(host="0.0.0.0", port=5000, debug=False)

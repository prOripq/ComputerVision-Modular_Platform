import json
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from core.video_buffer import VideoBuffer
from flask import send_from_directory
from modules.zone_module import ZoneDetector, save_zones, load_zones, Zone, TripLine

import cv2
import requests
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)

from auth import create_default_admin, verify_password
from core.database import DatabaseManager
from core.video_loader import VideoLoader
from modules.face_module import FaceRecognizer
from modules.person_module import PersonDetector

# ---------------------------------------------------------------------------
# Логгер
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

CONFIG_FILE = "config.json"

DEFAULT_CONFIG: dict = {
    "camera_source":       "0",
    "enable_face_rec":     True,
    "enable_people_count": True,
    "face_cooldown":       15.0,
    "log_interval":        5.0,
    "telegram_token":      "",
    "telegram_chat_id":    "",
    "enable_telegram":     False,
}

# ---------------------------------------------------------------------------
# Глобальное состояние
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
video_buffer: VideoBuffer | None = None

camera:          VideoLoader    | None = None
person_tracker:  PersonDetector | None = None
face_recognizer: FaceRecognizer | None = None
db:              DatabaseManager | None = None
zone_detector: ZoneDetector | None = None
config:          dict = {}
current_people_count: int = 0

_tg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tg_alert")


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
    global camera, person_tracker, face_recognizer, db, video_buffer

    create_default_admin()
    load_config()
    logger.info("--- ЗАГРУЗКА СИСТЕМЫ ---")

    src = config["camera_source"]
    src = int(src) if str(src).isdigit() else src

    camera          = VideoLoader(src)
    db              = DatabaseManager()
    person_tracker  = PersonDetector()
    face_recognizer = FaceRecognizer(db_path="known_faces")

    # Буфер: 10 сек до события + 10 сек после, клипы в папке clips/
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
<<<<<<< HEAD
        ret, buffer = cv2.imencode(".jpg", image_frame)
        if not ret:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            files={"photo": buffer.tobytes()},
            data={"chat_id": chat_id, "caption": message},
            timeout=10,
        )
    except Exception:
        logger.exception("Ошибка при отправке Telegram-уведомления.")


# ---------------------------------------------------------------------------
# Генератор кадров
# ---------------------------------------------------------------------------
=======
        ret, buffer = cv2.imencode('.jpg', image_frame)
        if not ret: return
        url = f"Telegram bot API"
        files = {'photo': buffer.tobytes()}
        data = {'chat_id': chat_id, 'caption': message}
        requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"Ошибка TG: {e}")
>>>>>>> d4bc81e1a3bb46e0a8be8cc1fe0ff014574d2b16

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

        if cfg.get("enable_people_count"):
            # detect() — только детекция, без рисования боксов YOLO
            track_ids, boxes_xywh, raw_plot = person_tracker.detect(frame)
 
            # Если есть активные зоны/линии — применяем их
            if zone_detector and (zone_detector.zones or zone_detector.lines):
                # Рисуем боксы YOLO на кадре
                frame = raw_plot
                # Поверх — зоны и линии
                frame, zone_stats = zone_detector.process(frame, track_ids, boxes_xywh)
                people_count = zone_stats["total"] if zone_stats["zones"] else len(track_ids)
 
                # Логируем статистику по каждой зоне отдельно
                if (time.time() - last_log_people_time) > float(cfg["log_interval"]):
                    for zs in zone_stats["zones"]:
                        db.log_event("ZONE_STATS", f"{zs['name']}:{zs['count']}")
                    for ls in zone_stats["lines"]:
                        db.log_event("LINE_CROSS", f"{ls['name']}:in={ls['in']},out={ls['out']}")
                    last_log_people_time = time.time()
            else:
                # Зон нет — старое поведение: просто рисуем боксы и считаем всех
                frame = person_tracker.draw_tails(raw_plot, track_ids, boxes_xywh)
                people_count = len(track_ids)
 
                if (time.time() - last_log_people_time) > float(cfg["log_interval"]):
                    db.log_event("PEOPLE_STATS", str(people_count))
                    last_log_people_time = time.time()
 
            with _state_lock:
                current_people_count = people_count

        if cfg.get("enable_face_rec"):
            frame, found_names = face_recognizer.recognize(frame)
            curr_time = time.time()

            for name in found_names:
                if (curr_time - last_seen_faces.get(name, 0.0)) > float(cfg["face_cooldown"]):
                    db.log_event("FACE_MATCH", name)
                    logger.info("Опознан: %s", name)
                    last_seen_faces[name] = curr_time

                    # ── Триггер записи видеоклипа ──
                    if video_buffer is not None:
                        video_buffer.trigger(label=f"face_{name}")

                    # ── Telegram с фото (как было) ──
                    if cfg.get("enable_telegram") and cfg.get("telegram_token"):
                        _tg_executor.submit(
                            _send_telegram_alert,
                            cfg["telegram_token"],
                            cfg["telegram_chat_id"],
                            f"🚨 ВНИМАНИЕ! Обнаружен: {name}",
                            frame.copy(),
                        )

        # ── Подаём каждый кадр в буфер (ОБЯЗАТЕЛЬНО после всей обработки) ──
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
# Роуты — аутентификация
# ---------------------------------------------------------------------------

@app.route("/zones")
@login_required
def zones_editor():
    return render_template("zones_editor.html")
 
 
@app.route("/api/zones", methods=["GET"])
@login_required
def api_zones_get():
    """Возвращает текущие зоны и линии."""
    zones_list, lines_list = load_zones()
    return jsonify({
        "zones": [z.__dict__ for z in zones_list],
        "lines": [l.__dict__ for l in lines_list],
    })
 
 
@app.route("/api/zones", methods=["POST"])
@login_required
def api_zones_post():
    """Сохраняет зоны и линии, перезагружает ZoneDetector."""
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
    """Сбрасывает счётчики пересечений линий."""
    if zone_detector:
        zone_detector.reset_counters()
    return jsonify({"status": "ok"})
 
 
@app.route("/snapshot")
@login_required
def snapshot():
    """Отдаёт один JPEG-кадр для редактора зон."""
    if camera is None:
        return "", 503
    frame = camera.get_frame()
    if frame is None:
        return "", 503
    ret, buf = cv2.imencode(".jpg", frame)
    if not ret:
        return "", 500
    from flask import Response
    return Response(buf.tobytes(), mimetype="image/jpeg")



@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authenticated"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if verify_password(username, password):
            session.permanent = True
            session["authenticated"] = True
            session["username"] = username
            logger.info("Успешный вход: %s", username)
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        else:
            logger.warning("Неудачная попытка входа: '%s'", username)
            error = "Неверный логин или пароль"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    session.clear()
    logger.info("Выход: %s", username)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Роуты — основные (все защищены @login_required)
# ---------------------------------------------------------------------------

@app.route("/clips")
@login_required
def list_clips():
    """Возвращает список сохранённых клипов."""
    clips_dir = "clips"
    if not os.path.exists(clips_dir):
        return jsonify([])
    files = sorted(
        [f for f in os.listdir(clips_dir) if f.endswith(".mp4")],
        reverse=True,
    )
    return jsonify(files)


@app.route("/clips/<path:filename>")
@login_required
def download_clip(filename):
    """Отдаёт файл клипа для скачивания/воспроизведения."""
    return send_from_directory(
        os.path.abspath("clips"),
        filename,
        as_attachment=False,  # False = можно воспроизводить прямо в браузере
    )


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

        save_config(new_config)

        if new_source != old_source:
            logger.info("Смена камеры: %s → %s", old_source, new_source)
            if camera:
                camera.release()
            src = int(new_source) if str(new_source).isdigit() else new_source
            camera = VideoLoader(src)

        return redirect(url_for("index"))

    return render_template("settings.html", config=config)


@app.route("/api/refresh_faces", methods=["POST"])
@login_required
def refresh_faces():
    try:
        face_recognizer.refresh_database()
        return jsonify({"status": "ok", "count": len(face_recognizer._names)})
    except Exception:
        logger.exception("Ошибка при обновлении базы лиц.")
        return jsonify({"error": "refresh_failed"}), 500
    
@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics")
@login_required
def api_analytics():
    try:
        period = int(request.args.get("days", 7))
        period = max(1, min(period, 90))  # Ограничиваем: от 1 до 90 дней

        return jsonify({
            "summary":      db.get_summary_stats(),
            "hourly":       db.get_hourly_stats(days=period),
            "daily":        db.get_daily_stats(days=period),
            "top_faces":    db.get_top_faces(limit=10),
        })
    except Exception:
        logger.exception("Ошибка при получении аналитики.")
        return jsonify({"error": "db_error"}), 500



# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_system()
<<<<<<< HEAD
    zone_detector = ZoneDetector()
    app.run(host="0.0.0.0", port=5000, debug=False)
=======
    app.run(host='0.0.0.0', port=5000, debug=False)
>>>>>>> d4bc81e1a3bb46e0a8be8cc1fe0ff014574d2b16

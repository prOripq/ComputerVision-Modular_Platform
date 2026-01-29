import json
import os
import threading
import requests
from flask import Flask, Response, request, redirect, url_for, render_template_string
import cv2
import time
from core.video_loader import VideoLoader
from core.database import DatabaseManager
from modules.face_module import FaceRecognizer
from modules.person_module import PersonDetector

app = Flask(__name__)
CONFIG_FILE = 'config.json'

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
camera = None
person_tracker = None
face_recognizer = None
db = None
config = {}
current_people_count = 0 

def load_config():
    global config
    if not os.path.exists(CONFIG_FILE):
        config = {
            "camera_source": "0",
            "enable_face_rec": True,
            "enable_people_count": True,
            "face_cooldown": 15.0,
            "log_interval": 5.0,
            "telegram_token": "",
            "telegram_chat_id": "",
            "enable_telegram": False
        }
    else:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

def save_config(new_config):
    global config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_config, f, indent=4)
    config = new_config

def init_system():
    global camera, person_tracker, face_recognizer, db, config
    load_config()
    print(f"--- ЗАГРУЗКА СИСТЕМЫ ---")
    src = config['camera_source']
    if src.isdigit(): src = int(src)
    if camera is None: camera = VideoLoader(src)
    if db is None: db = DatabaseManager()
    if person_tracker is None: person_tracker = PersonDetector()
    if face_recognizer is None: 
        face_recognizer = FaceRecognizer(db_path='known_faces')
        face_recognizer.app.prepare(ctx_id=0, det_size=(640, 640)) 

def send_telegram_alert(token, chat_id, message, image_frame):
    try:
        ret, buffer = cv2.imencode('.jpg', image_frame)
        if not ret: return
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {'photo': buffer.tobytes()}
        data = {'chat_id': chat_id, 'caption': message}
        requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"Ошибка TG: {e}")

def generate_frames():
    global config, current_people_count
    last_log_people_time = time.time()
    last_seen_faces = {}

    while True:
        if camera is None: 
            time.sleep(0.1)
            continue
        frame = camera.get_frame()
        if frame is None: break

        people_count = 0
        if config.get('enable_people_count'):
            frame, people_count = person_tracker.process(frame)
            current_people_count = people_count
            if (time.time() - last_log_people_time) > float(config['log_interval']):
                db.log_event("PEOPLE_STATS", str(people_count))
                last_log_people_time = time.time()

        if config.get('enable_face_rec'):
            frame, found_names = face_recognizer.process_and_return_names(frame)
            curr_time = time.time()
            for name in found_names:
                if name == "Unknown": continue
                last_time = last_seen_faces.get(name, 0)
                if (curr_time - last_time) > float(config['face_cooldown']):
                    db.log_event("FACE_MATCH", name)
                    last_seen_faces[name] = curr_time
                    if config.get('enable_telegram') and config.get('telegram_token'):
                        msg = f"🚨 ВНИМАНИЕ! Обнаружен: {name}"
                        threading.Thread(
                            target=send_telegram_alert, 
                            args=(config['telegram_token'], config['telegram_chat_id'], msg, frame)
                        ).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTING ---

@app.route('/')
def index():
    events = []
    last_face = "Нет данных"
    try:
        cur = db.connection.cursor()
        cur.execute("SELECT timestamp, event_type, details FROM events ORDER BY id DESC LIMIT 15")
        events = cur.fetchall()
        cur.execute("SELECT details FROM events WHERE event_type='FACE_MATCH' ORDER BY id DESC LIMIT 1")
        last_face_row = cur.fetchone()
        if last_face_row: last_face = last_face_row[0]
    except: pass
    return render_template_string(HTML_DASHBOARD, events=events, people_count=current_people_count, last_face=last_face)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global camera, config

    if request.method == 'POST':
        new_source = request.form.get('camera_source')
        new_config = config.copy()
        new_config['camera_source'] = new_source
        new_config['enable_face_rec'] = 'enable_face_rec' in request.form
        new_config['enable_people_count'] = 'enable_people_count' in request.form
        new_config['enable_telegram'] = 'enable_telegram' in request.form
        new_config['telegram_token'] = request.form.get('telegram_token', '').strip()
        new_config['telegram_chat_id'] = request.form.get('telegram_chat_id', '').strip()
        
        save_config(new_config)

        if new_source != config['camera_source']:
            if camera: camera.release()
            src = new_source
            if src.isdigit(): src = int(src)
            camera = VideoLoader(src)

        return redirect(url_for('index'))

    return render_template_string(HTML_SETTINGS, config=config)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- TEMPLATES ---

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>AI Security Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <meta http-equiv="refresh" content="3">
    <style>
        :root { --bg-dark: #0f1015; --panel-bg: #191b23; --accent-green: #00d25b; --accent-red: #fc424a; --text-main: #ffffff; --text-muted: #8e90a6; }
        body { background-color: var(--bg-dark); color: var(--text-main); font-family: 'Inter', sans-serif; }
        .navbar { background-color: var(--panel-bg); border-bottom: 1px solid #2c2e3e; padding: 15px 0; }
        .brand-logo { font-weight: 600; font-size: 1.2rem; color: var(--text-main); text-decoration: none; letter-spacing: 1px; }
        .stat-card { background-color: var(--panel-bg); border-radius: 12px; padding: 20px; display: flex; align-items: center; border: 1px solid #2c2e3e; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s; }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-icon { width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin-right: 15px; }
        .icon-blue { background: rgba(0, 144, 231, 0.2); color: #0090e7; }
        .icon-green { background: rgba(0, 210, 91, 0.2); color: #00d25b; }
        .icon-red { background: rgba(252, 66, 74, 0.2); color: #fc424a; }
        .video-container { background: #000; border-radius: 12px; overflow: hidden; position: relative; border: 1px solid #333; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
        .rec-indicator { position: absolute; top: 20px; right: 20px; display: flex; align-items: center; background: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
        .rec-dot { width: 10px; height: 10px; background-color: red; border-radius: 50%; margin-right: 8px; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0; } }
        .log-panel { background-color: var(--panel-bg); border-radius: 12px; height: 100%; border: 1px solid #2c2e3e; overflow: hidden; }
        .log-header { padding: 15px 20px; border-bottom: 1px solid #2c2e3e; font-weight: 600; }
        .log-list { max-height: 500px; overflow-y: auto; padding: 0; margin: 0; list-style: none; }
        .log-item { padding: 12px 20px; border-bottom: 1px solid #232530; display: flex; align-items: center; font-size: 0.9rem; }
        .log-item:last-child { border-bottom: none; }
        .log-time { font-size: 0.75rem; color: var(--text-muted); margin-right: 15px; min-width: 50px; }
        .log-badge { font-size: 0.7rem; padding: 4px 8px; border-radius: 4px; margin-right: 10px; font-weight: 600; text-transform: uppercase; }
        .badge-face { background: rgba(252, 66, 74, 0.2); color: #fc424a; }
        .badge-people { background: rgba(0, 210, 91, 0.2); color: #00d25b; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #191b23; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    </style>
</head>
<body>
    <nav class="navbar mb-4">
        <div class="container">
            <a href="#" class="brand-logo"><i class="fas fa-shield-alt text-primary me-2"></i>AI SECURITY GUARD</a>
            <div><a href="/settings" class="btn btn-outline-light btn-sm"><i class="fas fa-cog me-2"></i>Настройки</a></div>
        </div>
    </nav>
    <div class="container">
        <div class="row mb-4">
            <div class="col-md-4"><div class="stat-card"><div class="stat-icon icon-green"><i class="fas fa-walking"></i></div><div><div class="text-muted small">Людей в кадре</div><h3 class="mb-0">{{ people_count }}</h3></div></div></div>
            <div class="col-md-4"><div class="stat-card"><div class="stat-icon icon-red"><i class="fas fa-user-tag"></i></div><div><div class="text-muted small">Последний опознанный</div><h5 class="mb-0">{{ last_face }}</h5></div></div></div>
            <div class="col-md-4"><div class="stat-card"><div class="stat-icon icon-blue"><i class="fas fa-microchip"></i></div><div><div class="text-muted small">Статус системы</div><h5 class="mb-0 text-success">ONLINE (GPU)</h5></div></div></div>
        </div>
        <div class="row">
            <div class="col-lg-8 mb-4">
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" width="100%">
                    <div class="rec-indicator"><div class="rec-dot"></div> LIVE REC</div>
                </div>
            </div>
            <div class="col-lg-4 mb-4">
                <div class="log-panel">
                    <div class="log-header"><i class="fas fa-history me-2 text-muted"></i> Лента Событий</div>
                    <ul class="log-list">
                        {% for row in events %}
                        <li class="log-item">
                            <span class="log-time">{{ row[0].split(' ')[1] }}</span>
                            {% if row[1] == 'FACE_MATCH' %}<span class="log-badge badge-face">ALERT</span><span>Опознан: <strong>{{ row[2] }}</strong></span>
                            {% else %}<span class="log-badge badge-people">INFO</span><span>Людей: {{ row[2] }}</span>{% endif %}
                        </li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

HTML_SETTINGS = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Настройки системы</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg-dark: #0f1015; --panel-bg: #191b23; --accent-green: #00d25b; }
        body { background-color: var(--bg-dark); color: #fff; font-family: 'Inter', sans-serif; }
        
        /* Контейнер настроек */
        .settings-container { max-width: 800px; margin: 50px auto; }
        
        /* Заголовок секций - Сделали ярко-зеленым */
        .section-label {
            color: var(--accent-green); 
            text-transform: uppercase; letter-spacing: 1.2px; font-size: 0.8rem; 
            margin-bottom: 15px; font-weight: 700; display: block;
        }

        /* Инпуты */
        .custom-input { 
            background-color: #0f1015; border: 1px solid #2c2e3e; color: #fff; 
            padding: 12px 15px; border-radius: 8px; transition: 0.3s;
        }
        .custom-input:focus { 
            background-color: #0f1015; border-color: var(--accent-green); color: #fff; 
            box-shadow: 0 0 0 4px rgba(0, 210, 91, 0.1); outline: none;
        }
        .custom-input::placeholder { color: #555; }

        /* Подписи к полям - Сделали светло-серыми */
        .form-label { color: #e0e0e0; font-weight: 500; font-size: 0.9rem; }
        
        /* Описание под иконками - Сделали читаемым */
        .text-description { color: #adb5bd; font-size: 0.85rem; }

        /* Карточка модуля */
        .module-card {
            background-color: var(--panel-bg); border: 1px solid #2c2e3e; border-radius: 12px;
            padding: 20px; margin-bottom: 20px; transition: 0.3s;
        }
        .module-card:hover { border-color: #555; }
        
        /* Свичи (Тумблеры) */
        .form-switch .form-check-input {
            width: 3.5em; height: 1.8em; margin-top: 0;
            background-color: #2c2e3e; border-color: #444;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='-4 -4 8 8'%3e%3ccircle r='3' fill='%238e90a6'/%3e%3c/svg%3e");
        }
        .form-switch .form-check-input:checked {
            background-color: var(--accent-green); border-color: var(--accent-green);
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='-4 -4 8 8'%3e%3ccircle r='3' fill='%23fff'/%3e%3c/svg%3e");
        }

        /* Кнопки */
        .btn-save { 
            background-color: var(--accent-green); color: #000; font-weight: 600; 
            padding: 12px 30px; border-radius: 8px; border: none; 
        }
        .btn-save:hover { background-color: #00b34d; color: #000; }
        .btn-cancel { color: #adb5bd; text-decoration: none; margin-right: 20px; }
        .btn-cancel:hover { color: #fff; }

        .icon-box {
            width: 40px; height: 40px; border-radius: 8px; background: rgba(255,255,255,0.05);
            display: flex; align-items: center; justify-content: center; margin-right: 15px;
            color: var(--accent-green); font-size: 1.2rem;
        }
    </style>
</head>
<body>
    <div class="container settings-container">
        
        <div class="d-flex justify-content-between align-items-center mb-5">
            <h2 class="mb-0 text-white"><i class="fas fa-sliders-h me-3 text-primary"></i>Настройки Системы</h2>
            <a href="/" class="btn btn-outline-secondary btn-sm"><i class="fas fa-arrow-left me-2"></i>Назад</a>
        </div>

        <form method="POST">
            
            <span class="section-label">Источник Видео</span>
            <div class="module-card">
                <div class="mb-3">
                    <label class="form-label">RTSP ссылка или ID камеры</label>
                    <div class="input-group">
                        <span class="input-group-text" style="background: #2c2e3e; border-color: #2c2e3e; color: #aaa;"><i class="fas fa-video"></i></span>
                        <input type="text" class="form-control custom-input" name="camera_source" value="{{ config.camera_source }}" placeholder="Например: 0 или rtsp://admin:pass@192.168...">
                    </div>
                </div>
            </div>

            <span class="section-label mt-4">AI Модули</span>
            
            <div class="module-card d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center">
                    <div class="icon-box"><i class="fas fa-id-card-alt"></i></div>
                    <div>
                        <h5 class="mb-1 text-white">Распознавание Лиц</h5>
                        <div class="text-description">Идентификация сотрудников и нарушителей</div>
                    </div>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" name="enable_face_rec" {% if config.enable_face_rec %}checked{% endif %}>
                </div>
            </div>

            <div class="module-card d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center">
                    <div class="icon-box"><i class="fas fa-users"></i></div>
                    <div>
                        <h5 class="mb-1 text-white">Подсчет Людей</h5>
                        <div class="text-description">Статистика посетителей и трекинг</div>
                    </div>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" name="enable_people_count" {% if config.enable_people_count %}checked{% endif %}>
                </div>
            </div>

            <span class="section-label mt-4">Уведомления</span>
            <div class="module-card">
                <div class="d-flex align-items-center justify-content-between mb-4">
                    <div class="d-flex align-items-center">
                        <div class="icon-box" style="color: #0088cc;"><i class="fab fa-telegram-plane"></i></div>
                        <div>
                            <h5 class="mb-1 text-white">Telegram Бот</h5>
                            <div class="text-description">Отправка фото при обнаружении лиц</div>
                        </div>
                    </div>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" name="enable_telegram" {% if config.enable_telegram %}checked{% endif %}>
                    </div>
                </div>
                
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label">Bot Token</label>
                        <input type="text" class="form-control custom-input" name="telegram_token" value="{{ config.telegram_token }}" placeholder="12345:ABCde...">
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Chat ID</label>
                        <input type="text" class="form-control custom-input" name="telegram_chat_id" value="{{ config.telegram_chat_id }}" placeholder="123456789">
                    </div>
                </div>
            </div>

            <div class="mt-5 mb-5 d-flex justify-content-end align-items-center">
                <a href="/" class="btn-cancel">Отмена</a>
                <button type="submit" class="btn-save">Сохранить настройки</button>
            </div>

        </form>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    init_system()
    app.run(host='0.0.0.0', port=5000, debug=False)
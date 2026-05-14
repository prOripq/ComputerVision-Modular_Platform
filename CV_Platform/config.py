# Константы платформы
# Используется как справочник значений по умолчанию.
# Основная конфигурация хранится в config.json (создаётся автоматически).

DEFAULT_CAMERA_SOURCE = "0"
DEFAULT_FACE_COOLDOWN = 15.0      # секунд между повторными записями одного лица
DEFAULT_LOG_INTERVAL  = 5.0       # секунд между записями статистики людей
DEFAULT_CLEANUP_INTERVAL = 300.0  # секунд между чистками last_seen_faces

VIDEO_BUFFER_PRE_SECONDS  = 10.0
VIDEO_BUFFER_POST_SECONDS = 10.0
VIDEO_BUFFER_FPS          = 25.0
VIDEO_BUFFER_COOLDOWN     = 15.0

KNOWN_FACES_DIR = "known_faces"
CLIPS_DIR       = "clips"
ZONES_FILE      = "zones.json"
DB_FILE         = "platform_data.db"
CONFIG_FILE     = "config.json"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

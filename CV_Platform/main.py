import logging
import time

import cv2

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
# Настройки
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0
WINDOW_NAME = "CV Platform"
LOG_INTERVAL_PEOPLE = 5.0   # Писать кол-во людей раз в N секунд
FACE_COOLDOWN = 10.0        # Не писать одного человека чаще раза в N секунд
CLEANUP_INTERVAL = 300.0    # Чистить last_seen_faces раз в N секунд


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def run_platform() -> None:
    logger.info("--- ЗАПУСК ПЛАТФОРМЫ ---")

    loader = VideoLoader(VIDEO_SOURCE)
    db = DatabaseManager()
    person_tracker = PersonDetector()
    face_recognizer = FaceRecognizer(db_path="known_faces")

    last_log_people_time = time.time()
    last_cleanup_time = time.time()
    last_seen_faces: dict[str, float] = {}  # {имя: время последней записи}

    logger.info(">>> СИСТЕМА ЗАПИСЫВАЕТ ДАННЫЕ <<<")

    try:
        while True:
            frame = loader.get_frame()
            if frame is None:
                logger.warning("Кадр не получен — завершение.")
                break

            curr_time = time.time()

            # --- Подсчёт людей ---
            frame, people_count = person_tracker.process(frame)

            if (curr_time - last_log_people_time) > LOG_INTERVAL_PEOPLE:
                db.log_event("PEOPLE_STATS", str(people_count))
                logger.info("📊 Статистика: %d чел.", people_count)
                last_log_people_time = curr_time

            # --- Распознавание лиц ---
            frame, found_names = face_recognizer.recognize(frame)

            for name in found_names:
                last_time = last_seen_faces.get(name, 0.0)
                if (curr_time - last_time) > FACE_COOLDOWN:
                    db.log_event("FACE_MATCH", name)
                    logger.info("🚨 ОПОЗНАН: %s", name)
                    last_seen_faces[name] = curr_time

            # --- Чистка устаревших записей ---
            if (curr_time - last_cleanup_time) > CLEANUP_INTERVAL:
                cutoff = curr_time - FACE_COOLDOWN * 10
                last_seen_faces = {k: v for k, v in last_seen_faces.items() if v > cutoff}
                last_cleanup_time = curr_time

            # --- Отображение ---
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Выход по нажатию 'q'.")
                break

    except Exception:
        logger.exception("Необработанное исключение в основном цикле.")

    finally:
        loader.release()
        db.close()
        cv2.destroyAllWindows()
        logger.info("Ресурсы освобождены. Платформа остановлена.")


if __name__ == "__main__":
    run_platform()
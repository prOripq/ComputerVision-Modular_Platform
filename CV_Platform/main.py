import cv2
import time
from core.video_loader import VideoLoader
from core.database import DatabaseManager  # <-- НОВОЕ
from modules.face_module import FaceRecognizer
from modules.person_module import PersonDetector

# Настройки
VIDEO_SOURCE = 0 
WINDOW_NAME = "CV Platform: DB Connected"

# Настройки записи в БД
LOG_INTERVAL_PEOPLE = 5.0  # Писать кол-во людей раз в 5 секунд
FACE_COOLDOWN = 10.0       # Не писать одного и того же человека чаще чем раз в 10 сек

def run_platform():
    print("--- ЗАПУСК ПЛАТФОРМЫ ---")
    
    loader = VideoLoader(VIDEO_SOURCE)
    db = DatabaseManager() # <-- Подключаем базу
    
    person_tracker = PersonDetector()
    face_recognizer = FaceRecognizer(db_path='known_faces')
    
    # Переменные для таймеров
    last_log_people_time = time.time()
    last_seen_faces = {} # Словарь: {'Ivan': время_последней_записи}

    print(">>> СИСТЕМА ЗАПИСЫВАЕТ ДАННЫЕ <<<")

    while True:
        frame = loader.get_frame()
        if frame is None: break

        # 1. ОБРАБОТКА (Люди)
        frame, people_count = person_tracker.process(frame)
        
        # 2. ЛОГИКА ЗАПИСИ (Люди)
        curr_time = time.time()
        if (curr_time - last_log_people_time) > LOG_INTERVAL_PEOPLE:
            # Пишем в базу
            db.log_event("PEOPLE_STATS", str(people_count))
            print(f"📊 Статистика: {people_count} чел.")
            last_log_people_time = curr_time

        # 3. ОБРАБОТКА (Лица)
        # Нам нужно немного изменить логику вызова, чтобы получать имена, а не только картинку
        # Но так как process() у нас просто рисует, давай схитрим.
        # В идеале face_module должен возвращать список имен.
        # Пока сделаем так: face_recognizer.process изменяет frame, но мы добавим метод get_names?
        # Нет, давай проще: пусть process возвращает frame И список найденных имен.
        
        # ВНИМАНИЕ: Нам нужно зайти в face_module.py и чуть поправить его (см. шаг 4).
        # Предположим, мы это сделали, и он возвращает names
        frame, found_names = face_recognizer.process_and_return_names(frame)
        
        # 4. ЛОГИКА ЗАПИСИ (Лица)
        for name in found_names:
            if name == "Unknown": continue
            
            # Проверяем, когда видели его в последний раз
            last_time = last_seen_faces.get(name, 0)
            
            if (curr_time - last_time) > FACE_COOLDOWN:
                db.log_event("FACE_MATCH", name)
                print(f"🚨 ОПОЗНАН: {name}")
                last_seen_faces[name] = curr_time

        # Показываем
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    loader.release()
    db.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_platform()
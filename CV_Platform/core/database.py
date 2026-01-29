import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="platform_data.db"):
        self.connection = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        """Создаем таблицу событий, если её нет"""
        query = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT, 
            details TEXT
        )
        """
        self.cursor.execute(query)
        self.connection.commit()
        print("База данных подключена и проверена.")

    def log_event(self, event_type, details):
        """
        Записать событие.
        event_type: например 'FACE_DETECTED' или 'PEOPLE_COUNT'
        details: например 'Ivan' или '5'
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = "INSERT INTO events (timestamp, event_type, details) VALUES (?, ?, ?)"
        self.cursor.execute(query, (current_time, event_type, details))
        self.connection.commit()
        # print(f"[DB] Записано: {event_type} -> {details}") # Раскомментируй для отладки

    def close(self):
        self.connection.close()
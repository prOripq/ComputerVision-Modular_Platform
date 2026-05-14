import logging
import sqlite3
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Thread-safe обёртка над SQLite.
    Использует один коннект с RLock для защиты операций курсора.
    """

    def __init__(self, db_name: str = "platform_data.db"):
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(db_name, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._cursor = self.connection.cursor()
        self._create_tables()
        logger.info("База данных подключена: %s", db_name)

    def _create_tables(self) -> None:
        with self._lock:
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type  TEXT NOT NULL,
                    details     TEXT
                )
            """)
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events (event_type)
            """)
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events (timestamp)
            """)
            self.connection.commit()

    # ------------------------------------------------------------------
    # Базовые методы
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, details: str) -> None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._cursor.execute(
                "INSERT INTO events (timestamp, event_type, details) VALUES (?, ?, ?)",
                (current_time, event_type, details),
            )
            self.connection.commit()

    def get_recent_events(self, limit: int = 15) -> list[dict]:
        with self._lock:
            self._cursor.execute(
                """
                SELECT id, timestamp, event_type, details
                FROM events ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in self._cursor.fetchall()]

    def get_last_face_match(self) -> str | None:
        with self._lock:
            self._cursor.execute(
                """
                SELECT details FROM events
                WHERE event_type = 'FACE_MATCH'
                ORDER BY id DESC LIMIT 1
                """
            )
            row = self._cursor.fetchone()
            return row["details"] if row else None

    # ------------------------------------------------------------------
    # Аналитика
    # ------------------------------------------------------------------

    def get_hourly_stats(self, days: int = 7) -> list[dict]:
        with self._lock:
            self._cursor.execute(
                """
                SELECT
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    ROUND(AVG(CAST(details AS REAL)), 1)       AS avg_people
                FROM events
                WHERE
                    event_type = 'PEOPLE_STATS'
                    AND timestamp >= datetime('now', ? || ' days')
                GROUP BY hour
                ORDER BY hour
                """,
                (f"-{days}",),
            )
            rows = {row["hour"]: row["avg_people"] for row in self._cursor.fetchall()}
        return [{"hour": h, "avg_people": rows.get(h, 0)} for h in range(24)]

    def get_daily_stats(self, days: int = 30) -> list[dict]:
        with self._lock:
            self._cursor.execute(
                """
                SELECT
                    date(timestamp) AS date,
                    MAX(CAST(details AS REAL)) AS max_people
                FROM events
                WHERE
                    event_type = 'PEOPLE_STATS'
                    AND timestamp >= datetime('now', ? || ' days')
                GROUP BY date(timestamp)
                ORDER BY date
                """,
                (f"-{days}",),
            )
            people_by_day = {row["date"]: row["max_people"] for row in self._cursor.fetchall()}

            self._cursor.execute(
                """
                SELECT
                    date(timestamp) AS date,
                    COUNT(*)        AS face_events
                FROM events
                WHERE
                    event_type = 'FACE_MATCH'
                    AND timestamp >= datetime('now', ? || ' days')
                GROUP BY date(timestamp)
                ORDER BY date
                """,
                (f"-{days}",),
            )
            faces_by_day = {row["date"]: row["face_events"] for row in self._cursor.fetchall()}

        all_dates = sorted(set(list(people_by_day.keys()) + list(faces_by_day.keys())))
        return [
            {
                "date":        d,
                "max_people":  people_by_day.get(d, 0),
                "face_events": faces_by_day.get(d, 0),
            }
            for d in all_dates
        ]

    def get_top_faces(self, limit: int = 10) -> list[dict]:
        with self._lock:
            self._cursor.execute(
                """
                SELECT
                    details        AS name,
                    COUNT(*)       AS count,
                    MAX(timestamp) AS last_seen
                FROM events
                WHERE event_type = 'FACE_MATCH'
                GROUP BY details
                ORDER BY count DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in self._cursor.fetchall()]

    def get_summary_stats(self) -> dict:
        with self._lock:
            self._cursor.execute(
                "SELECT COUNT(*) AS cnt FROM events WHERE event_type = 'PEOPLE_STATS'"
            )
            total_observations = self._cursor.fetchone()["cnt"]

            self._cursor.execute(
                "SELECT COUNT(DISTINCT details) AS cnt FROM events WHERE event_type = 'FACE_MATCH'"
            )
            unique_faces = self._cursor.fetchone()["cnt"]

            self._cursor.execute(
                "SELECT COUNT(*) AS cnt FROM events WHERE event_type = 'FACE_MATCH'"
            )
            total_face_events = self._cursor.fetchone()["cnt"]

            self._cursor.execute(
                """
                SELECT
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    AVG(CAST(details AS REAL)) AS avg_people
                FROM events
                WHERE event_type = 'PEOPLE_STATS'
                GROUP BY hour
                ORDER BY avg_people DESC
                LIMIT 1
                """
            )
            peak_row = self._cursor.fetchone()
            peak_hour = f"{peak_row['hour']:02d}:00" if peak_row else "—"

            self._cursor.execute(
                "SELECT MAX(CAST(details AS REAL)) AS mx FROM events WHERE event_type = 'PEOPLE_STATS'"
            )
            max_row = self._cursor.fetchone()
            max_people = int(max_row["mx"]) if max_row and max_row["mx"] else 0

        return {
            "total_observations": total_observations,
            "unique_faces":       unique_faces,
            "total_face_events":  total_face_events,
            "peak_hour":          peak_hour,
            "max_people":         max_people,
        }

    def close(self) -> None:
        with self._lock:
            self.connection.close()
        logger.info("Соединение с БД закрыто.")
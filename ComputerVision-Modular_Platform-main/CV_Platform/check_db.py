from core.database import DatabaseManager

db = DatabaseManager()
print(f"{'ВРЕМЯ':<20} | {'СОБЫТИЕ':<15} | {'ДЕТАЛИ'}")
print("-" * 50)

# Берем последние 20 записей
db.cursor.execute("SELECT timestamp, event_type, details FROM events ORDER BY id DESC LIMIT 20")
rows = db.cursor.fetchall()

for row in rows:
    print(f"{row[0]:<20} | {row[1]:<15} | {row[2]}")

db.close()
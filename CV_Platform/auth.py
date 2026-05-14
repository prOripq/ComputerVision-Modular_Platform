import hashlib
import hmac
import json
import logging
import os
import secrets

logger = logging.getLogger(__name__)

USERS_FILE = "users.json"


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 — безопасно, без внешних зависимостей."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex()


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


def create_user(username: str, password: str) -> None:
    """
    Создаёт пользователя. Если файл users.json не существует — создаёт его.
    Вызывается один раз при первом запуске через create_default_admin().
    """
    users = _load_users()
    salt = secrets.token_hex(32)
    users[username] = {
        "salt":         salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    logger.info("Пользователь '%s' создан.", username)


def verify_password(username: str, password: str) -> bool:
    """Проверяет пару логин/пароль. Защищён от timing-атак через hmac.compare_digest."""
    users = _load_users()
    if username not in users:
        # Выполняем хэш вхолостую, чтобы время ответа не выдавало факт отсутствия юзера
        _hash_password(password, "dummy_salt")
        return False

    entry = users[username]
    expected = _hash_password(password, entry["salt"])
    return hmac.compare_digest(expected, entry["password_hash"])


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """
    Меняет пароль пользователя. Возвращает True при успехе.
    Требует подтверждения старого пароля.
    """
    if not verify_password(username, old_password):
        return False
    users = _load_users()
    if username not in users:
        return False
    salt = secrets.token_hex(32)
    users[username] = {
        "salt":          salt,
        "password_hash": _hash_password(new_password, salt),
    }
    _save_users(users)
    logger.info("Пароль изменён для пользователя '%s'.", username)
    return True


def create_default_admin(username: str = "admin", password: str = "admin") -> None:
    """
    Создаёт администратора по умолчанию при первом запуске,
    если файл users.json ещё не существует.
    """
    if os.path.exists(USERS_FILE):
        return
    logger.warning(
        "Файл users.json не найден. Создаётся пользователь '%s' с паролем '%s'. "
        "СМЕНИТЕ ПАРОЛЬ после первого входа!",
        username, password,
    )
    create_user(username, password)
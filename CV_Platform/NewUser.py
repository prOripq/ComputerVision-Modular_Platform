#!/usr/bin/env python3
"""
Утилита для создания пользователей платформы.

Использование:
    python NewUser.py <username>

Пароль будет запрошен интерактивно (не виден при вводе).
"""
import sys
import getpass

# Добавляем корневую папку в PATH
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import create_user, _load_users

def main():
    if len(sys.argv) < 2:
        print("Использование: python NewUser.py <username>")
        sys.exit(1)

    username = sys.argv[1].strip()
    if not username:
        print("Ошибка: имя пользователя не может быть пустым.")
        sys.exit(1)

    users = _load_users()
    if username in users:
        print(f"Пользователь '{username}' уже существует.")
        sys.exit(1)

    password = getpass.getpass(f"Пароль для '{username}': ")
    password2 = getpass.getpass("Повторите пароль: ")

    if password != password2:
        print("Ошибка: пароли не совпадают.")
        sys.exit(1)

    if len(password) < 6:
        print("Ошибка: пароль должен содержать не менее 6 символов.")
        sys.exit(1)

    create_user(username, password)
    print(f"✓ Пользователь '{username}' успешно создан.")

if __name__ == "__main__":
    main()
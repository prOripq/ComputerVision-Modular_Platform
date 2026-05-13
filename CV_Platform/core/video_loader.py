import logging
import os
import time

import cv2

# Должно быть установлено ДО первого обращения к VideoCapture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logger = logging.getLogger(__name__)


class VideoLoader:
    def __init__(
        self,
        source: int | str = 0,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 10,
        buffer_size: int = 1,
    ):
        """
        Args:
            source:                  ID камеры (int) или путь/URL (str).
            reconnect_delay:         Пауза между попытками переподключения (сек).
            max_reconnect_attempts:  Максимум попыток переподключения подряд.
                                     0 = бесконечно.
            buffer_size:             Размер буфера OpenCV. 1 = минимальная задержка
                                     для живых потоков.
        """
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.buffer_size = buffer_size

        # Файл — перематываем в начало; поток/камера — переподключаемся
        self._is_file = isinstance(source, str) and not source.startswith(("rtsp://", "rtmp://", "http://", "https://"))

        self.cap: cv2.VideoCapture | None = None
        self._connect()

    # ------------------------------------------------------------------
    # Подключение / переподключение
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Открывает источник. Возвращает True при успехе."""
        if self.cap is not None:
            self.cap.release()

        logger.info("Подключение к источнику: %s", self.source)
        
        # Проверяем, является ли источник веб-камерой (число или строка с числом)
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            # Для локальных камер используем стандартный бэкенд (cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            # Для RTSP/HTTP потоков и видеофайлов используем FFmpeg
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            logger.error("Не удалось открыть источник: %s", self.source)
            return False

        # Минимальный буфер — убирает нарастающую задержку на живых потоках
        if not self._is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        logger.info("Источник открыт успешно: %s", self.source)
        return True

    def _reconnect(self) -> bool:
        """
        Пытается переподключиться с паузами.
        Возвращает True если удалось, False если исчерпаны попытки.
        """
        attempt = 0
        while True:
            attempt += 1
            if self.max_reconnect_attempts > 0 and attempt > self.max_reconnect_attempts:
                logger.error(
                    "Превышено максимальное число попыток переподключения (%d).",
                    self.max_reconnect_attempts,
                )
                return False

            logger.warning(
                "Попытка переподключения %d/%s через %.1f сек...",
                attempt,
                self.max_reconnect_attempts or "∞",
                self.reconnect_delay,
            )
            time.sleep(self.reconnect_delay)

            if self._connect():
                logger.info("Переподключение успешно (попытка %d).", attempt)
                return True

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def get_frame(self):
        """
        Возвращает следующий кадр или None если источник недоступен.

        Для видеофайлов: при достижении конца перематывает в начало (loop).
        Для потоков/камер: при обрыве автоматически переподключается.
        """
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return None

        ret, frame = self.cap.read()

        if not ret:
            if self._is_file:
                # Файл закончился — перематываем
                logger.debug("Конец файла — перемотка в начало.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Не удалось прочитать кадр после перемотки.")
                    return None
            else:
                # Поток оборвался — переподключаемся
                logger.warning("Поток прерван. Попытка переподключения...")
                if not self._reconnect():
                    return None
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Не удалось прочитать кадр после переподключения.")
                    return None

        return frame

    def release(self) -> None:
        """Освобождает ресурсы."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("VideoLoader освобождён: %s", self.source)
import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoBuffer:
    """
    Кольцевой буфер кадров с записью клипа по событию.

    Принцип работы:
        - Постоянно хранит последние `pre_seconds` секунд в памяти (pre-buffer).
        - При вызове trigger() запускает запись ещё `post_seconds` секунд.
        - Итоговый клип = pre_buffer + post_buffer → файл .mp4.
        - Запись идёт в отдельном потоке, не блокируя основной цикл.
    """

    def __init__(
        self,
        output_dir:   str   = "clips",
        pre_seconds:  float = 10.0,
        post_seconds: float = 10.0,
        fps:          float = 25.0,
        resolution:   tuple[int, int] = (1280, 720),
        cooldown:     float = 15.0,
    ):
        """
        Args:
            output_dir:   Папка для сохранения клипов.
            pre_seconds:  Сколько секунд ДО события включать в клип.
            post_seconds: Сколько секунд ПОСЛЕ события включать в клип.
            fps:          Частота кадров исходного потока.
            resolution:   Разрешение (ширина, высота) для записи.
            cooldown:     Минимальная пауза между триггерами одного типа (сек).
        """
        self.output_dir   = output_dir
        self.pre_seconds  = pre_seconds
        self.post_seconds = post_seconds
        self.fps          = fps
        self.resolution   = resolution
        self.cooldown     = cooldown

        os.makedirs(output_dir, exist_ok=True)

        # Pre-буфер: maxlen = кол-во кадров за pre_seconds
        self._pre_buf: deque[np.ndarray] = deque(
            maxlen=int(pre_seconds * fps)
        )

        self._lock             = threading.Lock()
        self._recording        = False          # идёт ли сейчас запись post-части
        self._post_frames_left = 0              # сколько кадров ещё писать
        self._post_buf: list[np.ndarray] = []   # накапливаем post-кадры

        # {event_label: timestamp} — защита от спама триггеров
        self._last_trigger: dict[str, float] = {}

        # Очередь готовых клипов для сохранения в фоне
        self._save_queue: list[tuple[list, str]] = []
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="clip-saver"
        )
        self._save_thread.start()

        logger.info(
            "VideoBuffer готов: pre=%.1fs post=%.1fs fps=%.0f dir='%s'",
            pre_seconds, post_seconds, fps, output_dir,
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def push(self, frame: np.ndarray) -> None:
        """
        Подаёт очередной кадр в буфер.
        Вызывать на каждом кадре основного цикла.
        """
        with self._lock:
            # Всегда пишем в pre-буфер (deque сам выбрасывает старые кадры)
            self._pre_buf.append(frame.copy())

            # Если сейчас активна запись post-части — собираем кадры
            if self._recording:
                self._post_buf.append(frame.copy())
                self._post_frames_left -= 1

                if self._post_frames_left <= 0:
                    self._finish_recording()

    def trigger(self, label: str = "event") -> bool:
        """
        Запускает запись клипа вокруг текущего момента.

        Args:
            label: Метка события (используется в имени файла).
                   Также служит ключом для cooldown — одинаковые метки
                   не будут триггерить чаще чем раз в cooldown секунд.

        Returns:
            True  — запись запущена.
            False — проигнорировано (cooldown или уже пишем этот label).
        """
        now = time.time()

        with self._lock:
            last = self._last_trigger.get(label, 0.0)
            if now - last < self.cooldown:
                logger.debug("VideoBuffer: trigger '%s' проигнорирован (cooldown).", label)
                return False

            if self._recording:
                # Уже пишем — продлеваем post-часть если пришёл новый триггер
                self._post_frames_left = max(
                    self._post_frames_left,
                    int(self.post_seconds * self.fps),
                )
                logger.debug("VideoBuffer: trigger '%s' — продление записи.", label)
                return False

            self._last_trigger[label] = now
            self._recording = True
            self._post_frames_left = int(self.post_seconds * self.fps)
            self._post_buf = []

            # Снимаем слепок pre-буфера прямо сейчас
            pre_snapshot = list(self._pre_buf)
            clip_label = label
            # Передаём в очередь как «незавершённый» клип — post добавится в _finish
            self._pending_pre    = pre_snapshot
            self._pending_label  = clip_label

        logger.info("VideoBuffer: запись клипа по событию '%s'.", label)
        return True

    def release(self) -> None:
        """Завершает работу буфера. Ждёт сохранения незавершённых клипов."""
        logger.info("VideoBuffer: ожидание завершения записи клипов...")
        # Даём save_worker время доделать очередь
        for _ in range(30):
            with self._lock:
                if not self._save_queue:
                    break
            time.sleep(0.5)
        logger.info("VideoBuffer освобождён.")

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _finish_recording(self) -> None:
        """Вызывается под локом когда post_frames_left достиг 0."""
        pre  = self._pending_pre
        post = list(self._post_buf)
        label = self._pending_label

        self._recording = False
        self._post_buf  = []

        # Передаём в фоновый поток для записи на диск
        self._save_queue.append((pre + post, label))

        logger.info(
            "VideoBuffer: клип готов (%d кадров, label='%s'). Ставим в очередь записи.",
            len(pre) + len(post), label,
        )

    def _save_worker(self) -> None:
        """Фоновый поток: сохраняет клипы из очереди на диск."""
        while True:
            item = None
            with self._lock:
                if self._save_queue:
                    item = self._save_queue.pop(0)

            if item is None:
                time.sleep(0.2)
                continue

            frames, label = item
            self._write_clip(frames, label)

    def _write_clip(self, frames: list[np.ndarray], label: str) -> str | None:
        """Пишет список кадров в .mp4 файл. Возвращает путь или None при ошибке."""
        if not frames:
            logger.warning("VideoBuffer: пустой список кадров, клип не записан.")
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        filename = f"{safe_label}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        w, h = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.error("VideoBuffer: не удалось открыть VideoWriter для '%s'.", filepath)
            return None

        for frame in frames:
            # Приводим к нужному разрешению если отличается
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)

        writer.release()
        duration = len(frames) / self.fps
        logger.info(
            "VideoBuffer: клип сохранён → '%s' (%.1f сек, %d кадров).",
            filepath, duration, len(frames),
        )
        return filepath
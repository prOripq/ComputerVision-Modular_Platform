from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        tail_length: int = 30,
        tail_color: tuple = (230, 230, 230),
        tail_thickness: int = 2,
        stale_id_ttl: int = 60,
    ):
        """
        Args:
            model_path:     Путь к весам YOLO.
            tail_length:    Длина «хвоста» траектории в кадрах.
            tail_color:     BGR-цвет линии траектории.
            tail_thickness: Толщина линии траектории.
            stale_id_ttl:   Через сколько кадров удалять историю пропавшего ID.
        """
        print(f"Загрузка нейросети YOLO из '{model_path}'...")
        self.model = YOLO(model_path)

        self.tail_length = tail_length
        self.tail_color = tail_color
        self.tail_thickness = tail_thickness
        self.stale_id_ttl = stale_id_ttl

        # {track_id: deque([(x, y), ...])}
        self.track_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.tail_length)
        )
        # {track_id: кадр последнего появления}
        self._last_seen: dict[int, int] = {}
        self._frame_idx: int = 0

        print("Нейросеть готова к работе.")

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
        """
        Запускает трекинг и возвращает сырые данные без отрисовки.

        Returns:
            track_ids:  Список ID обнаруженных людей.
            boxes_xywh: Массив боксов (N, 4) в формате xywh.
            raw_plot:   Кадр с боксами от YOLO (без наших хвостов).
        """
        results = self.model.track(frame, persist=True, classes=0, verbose=False)
        result = results[0]

        raw_plot = result.plot()

        if result.boxes is None or result.boxes.id is None:
            return [], np.empty((0, 4)), raw_plot

        track_ids = result.boxes.id.int().cpu().tolist()
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        return track_ids, boxes_xywh, raw_plot

    def draw_tails(
        self,
        frame: np.ndarray,
        track_ids: list[int],
        boxes_xywh: np.ndarray,
    ) -> np.ndarray:
        """
        Обновляет историю траекторий и рисует «хвосты» на кадре.

        Returns:
            Кадр с нарисованными хвостами.
        """
        annotated = frame.copy()

        for track_id, box in zip(track_ids, boxes_xywh):
            x, y, _, h = box
            foot_point = (float(x), float(y + h / 2))

            track = self.track_history[track_id]
            track.append(foot_point)
            self._last_seen[track_id] = self._frame_idx

            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    annotated,
                    [points],
                    isClosed=False,
                    color=self.tail_color,
                    thickness=self.tail_thickness,
                )

        return annotated

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Удобный метод «всё в одном»: детекция + отрисовка.

        Returns:
            annotated_frame: Кадр с боксами YOLO и хвостами траекторий.
            people_count:    Количество людей в текущем кадре.
        """
        self._frame_idx += 1

        track_ids, boxes_xywh, raw_plot = self.detect(frame)
        annotated_frame = self.draw_tails(raw_plot, track_ids, boxes_xywh)

        self._cleanup_stale_ids()

        return annotated_frame, len(track_ids)

    def reset(self) -> None:
        """Сбрасывает всю историю трекинга (например, при смене видео)."""
        self.track_history.clear()
        self._last_seen.clear()
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _cleanup_stale_ids(self) -> None:
        """Удаляет историю ID, которые не появлялись дольше stale_id_ttl кадров."""
        stale = [
            tid
            for tid, last in self._last_seen.items()
            if self._frame_idx - last > self.stale_id_ttl
        ]
        for tid in stale:
            del self.track_history[tid]
            del self._last_seen[tid]
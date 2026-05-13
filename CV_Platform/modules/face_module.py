import cv2
import numpy as np
import os
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(
        self,
        db_path: str = "known_faces",
        similarity_threshold: float = 0.55,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int | None = None,
    ):
        """
        Args:
            db_path:              Путь к папке с фотографиями известных лиц.
            similarity_threshold: Порог cosine similarity для распознавания (0.0–1.0).
                                  Рекомендуется 0.50–0.65. Ниже — больше ложных совпадений.
            det_size:             Размер входа детектора YOLO внутри InsightFace.
            ctx_id:               ID устройства: 0+ = GPU, -1 = CPU.
                                  None = автодетект (GPU если доступен, иначе CPU).
        """
        if ctx_id is None:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            except ImportError:
                ctx_id = -1

        device_label = f"GPU (ctx_id={ctx_id})" if ctx_id >= 0 else "CPU"
        print(f"Загрузка InsightFace на {device_label}...")

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        self.db_path = db_path
        self.similarity_threshold = similarity_threshold

        # Хранится как матрица (N, embedding_dim) для быстрого np.dot
        self._embeddings_matrix: np.ndarray = np.empty((0,))
        self._names: list[str] = []

        self.load_database()

    # ------------------------------------------------------------------
    # База данных
    # ------------------------------------------------------------------

    def load_database(self) -> None:
        """Читает папку db_path и строит матрицу эмбеддингов."""
        embeddings: list[np.ndarray] = []
        names: list[str] = []

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            logger.warning("Папка '%s' не найдена — создана пустая.", self.db_path)
            self._embeddings_matrix = np.empty((0,))
            self._names = []
            return

        print(f"Загрузка базы лиц из '{self.db_path}'...")

        for filename in os.listdir(self.db_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            filepath = os.path.join(self.db_path, filename)
            img = cv2.imread(filepath)

            if img is None:
                logger.warning("Не удалось прочитать файл: %s — пропущен.", filepath)
                continue

            faces = self.app.get(img)
            name = os.path.splitext(filename)[0]

            if len(faces) == 0:
                logger.warning("Лицо не найдено в файле: %s — пропущен.", filename)
                continue

            if len(faces) > 1:
                logger.warning(
                    "В файле '%s' найдено %d лиц — используется первое.",
                    filename,
                    len(faces),
                )

            embeddings.append(faces[0].normed_embedding)
            names.append(name)

        self._names = names
        self._embeddings_matrix = (
            np.stack(embeddings) if embeddings else np.empty((0,))
        )

        print(f"База готова. Людей в базе: {len(self._names)}")

    def refresh_database(self) -> None:
        """Перезагружает базу (например, после добавления новых фото через веб-интерфейс)."""
        self.load_database()

    # ------------------------------------------------------------------
    # Распознавание
    # ------------------------------------------------------------------

    def recognize(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Находит и распознаёт лица на кадре.

        Args:
            frame: Кадр BGR (не мутируется).

        Returns:
            annotated_frame: Копия кадра с нарисованными боксами и подписями.
            found_names:     Имена распознанных людей (без «Unknown»).
        """
        annotated = frame.copy()
        faces = self.app.get(annotated)
        found_names: list[str] = []

        for face in faces:
            box = face.bbox.astype(int)
            name, score = self._match(face.normed_embedding)

            if name is not None:
                label = f"{name} ({int(score * 100)}%)"
                color = (0, 255, 0)   # зелёный — найден
                found_names.append(name)
            else:
                label = "Unknown"
                color = (0, 0, 255)   # красный — не найден

            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                annotated,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        return annotated, found_names

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _match(
        self, embedding: np.ndarray
    ) -> tuple[str | None, float | None]:
        
        if len(self._names) == 0:
            return None, None

        # Матричное умножение: (N, D) · (D,) → (N,)
        similarities = self._embeddings_matrix @ embedding
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            return self._names[best_idx], best_score

        return None, None
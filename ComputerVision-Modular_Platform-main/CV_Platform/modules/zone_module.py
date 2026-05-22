import json
import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ищем системный шрифт с поддержкой кириллицы
_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",           # Windows
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
]

def _find_font(size: int = 18) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

logger = logging.getLogger(__name__)

ZONES_FILE = "zones.json"


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """Полигональная зона — считаем людей внутри."""
    id:     str
    name:   str
    points: list[list[int]]   # [[x, y], [x, y], ...]
    color:  list[int] = field(default_factory=lambda: [0, 210, 255])  # BGR


@dataclass
class TripLine:
    """Линия пересечения — считаем входы/выходы."""
    id:     str
    name:   str
    p1:     list[int]          # [x, y]
    p2:     list[int]          # [x, y]
    color:  list[int] = field(default_factory=lambda: [0, 100, 255])  # BGR


# ---------------------------------------------------------------------------
# Загрузка / сохранение
# ---------------------------------------------------------------------------

def load_zones() -> tuple[list[Zone], list[TripLine]]:
    if not os.path.exists(ZONES_FILE):
        return [], []
    try:
        with open(ZONES_FILE, "r") as f:
            data = json.load(f)
        zones = [Zone(**z) for z in data.get("zones", [])]
        lines = [TripLine(**l) for l in data.get("lines", [])]
        logger.info("Загружено зон: %d, линий: %d", len(zones), len(lines))
        return zones, lines
    except Exception:
        logger.exception("Ошибка загрузки zones.json")
        return [], []


def save_zones(zones: list[Zone], lines: list[TripLine]) -> None:
    data = {
        "zones": [z.__dict__ for z in zones],
        "lines": [l.__dict__ for l in lines],
    }
    with open(ZONES_FILE, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Зоны сохранены: %d зон, %d линий", len(zones), len(lines))


# ---------------------------------------------------------------------------
# Главный класс
# ---------------------------------------------------------------------------

class ZoneDetector:
    """
    Принимает треки от PersonDetector и:
      - считает людей внутри каждой зоны (полигон)
      - считает пересечения каждой линии (вход / выход)
    """

    def __init__(self):
        self.zones:  list[Zone]     = []
        self.lines:  list[TripLine] = []

        # Счётчики пересечений линий: {line_id: {"in": N, "out": N}}
        self.line_counters: dict[str, dict[str, int]] = {}

        # Предыдущая сторона линии для каждого трека: {line_id: {track_id: side}}
        # side = +1 или -1 (знак cross-product)
        self._prev_side: dict[str, dict[int, int]] = {}

        self.load()

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------

    def load(self) -> None:
        self.zones, self.lines = load_zones()
        for line in self.lines:
            if line.id not in self.line_counters:
                self.line_counters[line.id] = {"in": 0, "out": 0}
            if line.id not in self._prev_side:
                self._prev_side[line.id] = {}

    def reload(self) -> None:
        """Перезагружает конфигурацию зон без сброса счётчиков."""
        zones, lines = load_zones()
        self.zones = zones

        # Добавляем новые линии, не удаляя счётчики существующих
        existing_ids = {l.id for l in self.lines}
        for line in lines:
            if line.id not in existing_ids:
                self.line_counters[line.id] = {"in": 0, "out": 0}
                self._prev_side[line.id] = {}
        self.lines = lines

    def reset_counters(self) -> None:
        """Сбрасывает счётчики пересечений."""
        for lid in self.line_counters:
            self.line_counters[lid] = {"in": 0, "out": 0}
        self._prev_side = {lid: {} for lid in self._prev_side}

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        track_ids: list[int],
        boxes_xywh: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Обрабатывает кадр с треками.

        Args:
            frame:      BGR-кадр (будет изменён — рисуются зоны и линии).
            track_ids:  Список ID треков от PersonDetector.
            boxes_xywh: Боксы (N, 4) в формате xywh.

        Returns:
            annotated_frame: Кадр с нарисованными зонами, линиями и счётчиками.
            stats: {
                "zones":  [{"id": ..., "name": ..., "count": N}, ...],
                "lines":  [{"id": ..., "name": ..., "in": N, "out": N}, ...],
                "total":  N,   # людей во всех зонах (union)
            }
        """
        annotated = frame.copy()

        # Центры "ног" для каждого трека
        foot_points: dict[int, tuple[float, float]] = {}
        for tid, box in zip(track_ids, boxes_xywh):
            x, y, _, h = box
            foot_points[tid] = (float(x), float(y + h / 2))

        # ── Зоны ──
        zone_stats = []
        zone_people: set[int] = set()   # union по всем зонам

        for zone in self.zones:
            pts = np.array(zone.points, dtype=np.int32)
            color = tuple(zone.color)

            # Полупрозрачная заливка
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

            # Контур зоны
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Считаем людей внутри
            inside_ids = [
                tid for tid, (fx, fy) in foot_points.items()
                if cv2.pointPolygonTest(pts, (fx, fy), False) >= 0
            ]
            zone_people.update(inside_ids)

            # Подпись зоны
            label = f"{zone.name}: {len(inside_ids)} person(s)"
            cx = int(np.mean([p[0] for p in zone.points]))
            cy = int(np.mean([p[1] for p in zone.points]))
            _draw_label(annotated, label, (cx, cy), color)

            zone_stats.append({"id": zone.id, "name": zone.name, "count": len(inside_ids)})

        # ── Линии ──
        line_stats = []

        for line in self.lines:
            p1 = tuple(line.p1)
            p2 = tuple(line.p2)
            color = tuple(line.color)

            cv2.line(annotated, p1, p2, color, 2)

            # Стрелка — показывает направление «вход»
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Нормаль к линии (перпендикуляр, указывает «внутрь»)
            nx, ny = -dy, dx
            norm = max((nx**2 + ny**2) ** 0.5, 1e-6)
            arrow_end = (int(mid[0] + nx / norm * 20), int(mid[1] + ny / norm * 20))
            cv2.arrowedLine(annotated, mid, arrow_end, color, 2, tipLength=0.4)

            # Считаем пересечения
            in_count, out_count = self._update_line_crossings(
                line, foot_points, track_ids
            )

            cnts = self.line_counters[line.id]
            label = f"{line.name}  IN:{cnts['in']} OUT:{cnts['out']}"
            _draw_label(annotated, label, (p1[0], p1[1] - 12), color)

            line_stats.append({
                "id":   line.id,
                "name": line.name,
                "in":   cnts["in"],
                "out":  cnts["out"],
            })

        stats = {
            "zones": zone_stats,
            "lines": line_stats,
            "total": len(zone_people),
        }
        return annotated, stats

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _update_line_crossings(
        self,
        line: TripLine,
        foot_points: dict[int, tuple[float, float]],
        track_ids: list[int],
    ) -> tuple[int, int]:
        """
        Определяет пересечения линии по смене знака cross-product.

        Cross-product (p1→p2) × (p1→foot) > 0  →  сторона +1
        Cross-product (p1→p2) × (p1→foot) < 0  →  сторона -1

        Смена знака = пересечение. Направление смены = вход/выход.
        """
        x1, y1 = line.p1
        x2, y2 = line.p2
        prev = self._prev_side[line.id]

        new_in = new_out = 0

        # Чистим треки которых больше нет
        active = set(track_ids)
        stale = [tid for tid in prev if tid not in active]
        for tid in stale:
            del prev[tid]

        for tid, (fx, fy) in foot_points.items():
            # Cross product: (p2-p1) × (foot-p1)
            cross = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1)
            side = 1 if cross >= 0 else -1

            if tid in prev and prev[tid] != side:
                # Пересечение!
                if side == 1:
                    self.line_counters[line.id]["in"] += 1
                    new_in += 1
                else:
                    self.line_counters[line.id]["out"] += 1
                    new_out += 1

            prev[tid] = side

        return new_in, new_out


# ---------------------------------------------------------------------------
# Утилита рисования
# ---------------------------------------------------------------------------

def _draw_label(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Рисует подпись с тёмным фоном для читаемости."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    y = max(y, th + 4)
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
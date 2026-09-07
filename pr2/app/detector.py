"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.
"""

import io
import time
from typing import Any, Dict, List, Optional
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

# Глобальна змінна для кешування завантаженої моделі
_MODEL_INSTANCE: Optional[YOLO] = None


class DetectionError(Exception):
    """Базова помилка детекції, зрозуміла веб-рівню."""
    pass


class InvalidImageError(DetectionError):
    """Помилка некоректного файлу (не зображення або порожній)."""
    pass


class ModelInferenceError(DetectionError):
    """Помилка під час виконання інференсу моделлю."""
    pass


def load_model() -> YOLO:
    """Повернути готову до роботи модель.

    Завантаження ваг здійснюється один раз при першому виклику й кешується.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = YOLO(WEIGHTS)
    return _MODEL_INSTANCE


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Any]:
    """Знайти обʼєкти на зображенні.

    Приймає байти завантаженого файлу, повертає структурований результат.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise InvalidImageError("Файл порожній.")

    # Відкриваємо зображення та конвертуємо в RGB (усуває проблеми з PNG/RGBA та дескрипторами файлів)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise InvalidImageError("Наданий файл не є валідним зображенням.")

    model = load_model()

    # Вимірюємо безпосередній час інференсу моделі
    start_time = time.perf_counter()
    try:
        results = model.predict(source=img, conf=confidence, verbose=False)
    except Exception as e:
        raise ModelInferenceError(f"Помилка під час виклику моделі: {str(e)}")
    
    inference_time = time.perf_counter() - start_time

    detected_objects: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names.get(class_id, "unknown")
            conf = round(float(box.conf[0].item()), 4)

            # Отримуємо рамку [xmin, ymin, xmax, ymax]
            xyxy = box.xyxy[0].tolist()
            bbox = [round(coord, 2) for coord in xyxy]

            detected_objects.append({
                "class": class_name,
                "confidence": conf,
                "bbox": bbox
            })

    return {
        "count": len(detected_objects),
        "inference_time_ms": round(inference_time * 1000, 2),
        "confidence_threshold": confidence,
        "objects": detected_objects
    }
"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

from . import detector

app = FastAPI(title="Детекція обʼєктів — ПР2")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.on_event("startup")
def startup_event():
    """Прогрів (завантаження) моделі при старті сервера."""
    detector.load_model()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=detector.DEFAULT_CONFIDENCE)
):
    """Повернути знайдені на зображенні обʼєкти у форматі JSON."""
    content = await image.read()

    try:
        result = detector.detect(content, confidence=confidence)
        return JSONResponse(content=result)
    except detector.InvalidImageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except detector.ModelInferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except detector.DetectionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка детекції: {str(e)}"
        )
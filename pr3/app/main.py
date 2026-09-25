"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит до неї, — усе це лишається в `app/llm.py`. Тут
вирішується інше: що застосунок віддає клієнтові та з яким HTTP-статусом.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import llm

# Налаштування логування технічних помилок для розробника
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    """Звернення користувача."""

    question: str


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель."""
    if not CONTEXT_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл контексту context.md не знайдено."
        )
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON з обробкою помилок."""
    # Валідація: перевірка на порожній запит або пробіли
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Будь ласка, введіть Ваше запитання."
        )

    try:
        context = load_context()
        return llm.ask(clean_question, context)

    except llm.LLMError as e:
        # Логуємо деталі помилки для сервера, але віддаємо користувачеві зрозуміле повідомлення
        logger.error(f"LLMError [{e.status_code}]: {e.message}")
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.message}
        )

    except Exception as e:
        logger.exception("Непередбачувана помилка сервера")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Внутрішня помилка сервера. Спробуйте пізніше."}
        )
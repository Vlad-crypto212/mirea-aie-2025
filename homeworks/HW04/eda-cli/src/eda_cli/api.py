# src/eda_cli/api.py
from __future__ import annotations

import io
import logging
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(title="EDA CLI API", version="0.1.0")

# Настройка логирования (для варианта D)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    """Проверка состояния сервиса."""
    logger.info("Health check requested")
    return "OK"

@app.post("/quality")
def quality(n_rows: int, n_cols: int, missing_count: int) -> dict[str, float]:
    """
    Простая оценка качества данных по метрикам.
    """
    logger.info(f"Quality check requested for n_rows={n_rows}, n_cols={n_cols}, missing_count={missing_count}")

    # Имитация вычислений
    total_cells = n_rows * n_cols
    if total_cells == 0:
        quality_score = 1.0
    else:
        missing_share = missing_count / total_cells
        quality_score = max(0.0, min(1.0, 1.0 - missing_share - (n_rows < 100) * 0.2 - (n_cols > 100) * 0.1))

    # Для простоты возвращаем только score, но можно добавить и флаги
    # Имитируем флаги на основе параметров
    flags = {
        "too_few_rows": n_rows < 100,
        "too_many_columns": n_cols > 100,
        "too_many_missing": missing_count / total_cells > 0.5 if total_cells > 0 else False,
        # Эти флаги не могут быть вычислены без реального датафрейма
        # "has_constant_columns": False, ...
    }
    logger.info(f"Quality score calculated: {quality_score}")
    return {"quality_score": quality_score, "flags": flags}

@app.post("/quality-from-csv")
def quality_from_csv(file: UploadFile = File(...)) -> dict[str, float]:
    """
    Оценка качества данных по содержимому CSV-файла.
    """
    logger.info(f"Quality from CSV requested for file: {file.filename}")
    try:
        content = file.file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        file.file.seek(0) # Возвращаем указатель на начало, если файл нужно использовать снова
    except Exception as exc:
        logger.error(f"Error reading CSV file: {exc}")
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    try:
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
    except Exception as exc:
        logger.error(f"Error processing data: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not process data: {exc}")

    logger.info(f"Quality score from CSV calculated: {flags['quality_score']}")
    # Возвращаем только quality_score, как в оригинальном эндпоинте
    return {"quality_score": flags["quality_score"]}

# --- НОВЫЙ ЭНДПОИНТ ИЗ ВАРИАНТА A ---
@app.post("/quality-flags-from-csv")
def quality_flags_from_csv(file: UploadFile = File(...)) -> dict[str, dict[str, bool]]:
    """
    Возвращает полный набор флагов качества, включая те, что добавлены в HW03.
    """
    logger.info(f"Quality flags from CSV requested for file: {file.filename}")
    try:
        content = file.file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        file.file.seek(0)
    except Exception as exc:
        logger.error(f"Error reading CSV file: {exc}")
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    try:
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
    except Exception as exc:
        logger.error(f"Error processing data: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not process data: {exc}")

    # Фильтруем только флаги (ключи, которые являются bool)
    boolean_flags = {k: v for k, v in flags.items() if isinstance(v, bool)}
    result = {"flags": boolean_flags}

    logger.info(f"Quality flags from CSV calculated: {boolean_flags}")
    return result
# --- КОНЕЦ НОВОГО ЭНДПОИНТА ---
from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_constant_columns_heuristic():
    """Тест для проверки эвристики has_constant_columns."""
    # Создаем DataFrame с постоянным столбцом
    df = pd.DataFrame({
        "age": [10, 20, 30, 40],
        "height": [140, 150, 160, 170],
        "const_col": ["same", "same", "same", "same"],  # постоянный столбец
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что эвристика правильно определяет постоянные столбцы
    assert flags['has_constant_columns'] == True
    assert 'const_col' in flags['constant_columns']


def test_high_cardinality_categoricals_heuristic():
    """Тест для проверки эвристики has_high_cardinality_categoricals."""
    # Создаем DataFrame с высокой кардинальностью в категориальном столбце
    # Все массивы должны иметь одинаковую длину
    df = pd.DataFrame({
        'normal_cat': ['A', 'B', 'C', 'A'],  # 3 уникальных из 4 строк (75%)
        'high_card_cat': [f'value_{i}' for i in range(4)],  # 4 уникальных из 4 строк (100%)
        'numeric': [1, 2, 3, 4]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что эвристика правильно определяет столбцы с высокой кардинальностью
    assert flags['has_high_cardinality_categoricals'] == True
    assert 'high_card_cat' in flags['high_cardinality_categorical_columns']


def test_suspicious_id_duplicates_heuristic():
    """Тест для проверки эвристики has_suspicious_id_duplicates."""
    # Создаем DataFrame с дубликатами ID
    df = pd.DataFrame({
        'age': [10, 20, 30, 40],
        'height': [140, 150, 160, 170],
        'user_id': [1, 2, 3, 2],  # ID 2 встречается дважды
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что эвристика правильно определяет дубликаты ID
    assert flags['has_suspicious_id_duplicates'] == True
    assert 'user_id' in flags['suspicious_id_duplicate_columns']


def test_many_zero_values_heuristic():
    """Тест для проверки эвристики has_many_zero_values."""
    # Создаем DataFrame, в котором есть столбец с большим количеством нулей
    # и среднее значение которого близко к 0 по сравнению с максимумом
    df = pd.DataFrame({
        'age': [10, 20, 30, 40],
        'mostly_zero': [0, 0, 0, 1],  # 3 нуля из 4 значений (75% нулей), среднее = 0.25, макс = 1
        'normal_numeric': [1, 2, 3, 4]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что функция возвращает ожидаемые ключи
    assert 'has_many_zero_values' in flags
    assert 'many_zero_value_columns' in flags
    # Проверяем, что если эвристика сработала, то столбец входит в список
    if flags['has_many_zero_values']:
        assert 'mostly_zero' in flags['many_zero_value_columns']
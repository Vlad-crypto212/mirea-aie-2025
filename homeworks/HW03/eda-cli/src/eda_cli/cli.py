from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(10, help="Количество топ-категорий для отображения."),
    title: str = typer.Option("EDA Report", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.5, help="Минимальная доля пропусков, при которой колонка считается проблемной."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)  # передаем top_k_categories в функцию

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")
        
        # Добавляем информацию о новых эвристиках
        f.write(f"- Постоянные столбцы: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Категориальные признаки с высокой кардинальностью: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        f.write(f"- Подозрительные дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n")
        f.write(f"- Много нулевых значений: **{quality_flags['has_many_zero_values']}**\n\n")
        
        # Если есть постоянные столбцы, перечисляем их
        if quality_flags['has_constant_columns']:
            f.write(f"  - Постоянные столбцы: {', '.join(quality_flags['constant_columns'])}\n")
        
        # Если есть признаки с высокой кардинальностью, перечисляем их
        if quality_flags['has_high_cardinality_categoricals']:
            f.write(f"  - Признаки с высокой кардинальностью: {', '.join(quality_flags['high_cardinality_categorical_columns'])}\n")
        
        # Если есть подозрительные дубликаты ID, перечисляем их
        if quality_flags['has_suspicious_id_duplicates']:
            f.write(f"  - Столбцы с дубликатами ID: {', '.join(quality_flags['suspicious_id_duplicate_columns'])}\n")
        
        # Если есть столбцы с большим количеством нулей, перечисляем их
        if quality_flags['has_many_zero_values']:
            f.write(f"  - Столбцы с большим количеством нулей: {', '.join(quality_flags['many_zero_value_columns'])}\n\n")
        
        # Добавляем информацию о параметрах отчета
        f.write("## Параметры отчета\n\n")
        f.write(f"- Максимальное количество гистограмм: **{max_hist_columns}**\n")
        f.write(f"- Количество топ-категорий: **{top_k_categories}**\n")
        f.write(f"- Порог доли пропусков: **{min_missing_share:.2%}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            # Правильный способ получить проблемные столбцы из missing_df
            # missing_df использует названия столбцов как индекс, а не как отдельный столбец 'column'
            problematic_mask = missing_df['missing_share'] >= min_missing_share
            problematic_cols = missing_df.index[problematic_mask].tolist()  # Получаем индекс (названия столбцов)
            if problematic_cols:
                f.write(f"**Проблемные столбцы (доля пропусков ≥ {min_missing_share:.2%}):** {', '.join(problematic_cols)}\n\n")
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
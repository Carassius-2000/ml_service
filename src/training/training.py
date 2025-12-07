"""
Модуль для обновления модели предсказания цены бриллиантов.

Основная функциональность:
1. Загрузка данных из SQLite-базы.
2. Предобработка данных (очистка, преобразование типов, разделение признаков).
3. Загрузка текущей версии модели и её метрики качества из реестра.
4. Оценка качества модели на новых данных с использованием кросс-валидации.
5. Обновление реестра моделей при улучшении метрики качества.
"""

import json
import sqlite3
import sys
from datetime import date

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

from loggers import error_logger, info_logger
from processing import change_types, clean_data, rename_columns, split_data

MODEL_REGISTRY_PATH = "model_registry"
MODEL_SETTINGS_PATH = f"{MODEL_REGISTRY_PATH}/modelsettings.json"


def load_data() -> pd.DataFrame:
    """
    Загружает данные о бриллиантах из SQLite.

    Returns
    -------
    `pd.DataFrame`
        Набор данных, необходимый для обучения модели.

    Raises
    ------
    `sqlite3.OperationalError`
        Возникает если файл базы данных не существует.
    `pandas.errors.DatabaseError`
        Возникает если таблицы в базе данных нет.
    """
    try:
        with sqlite3.connect("file:diamonds.db?mode=ro", uri=True) as connection:
            query = "SELECT carat, cut, color, clarity, price FROM diamonds"
            df = pd.read_sql(query, connection)
        info_logger.info("Данные успешно загружены из БД.")
        return df
    except sqlite3.OperationalError as e:
        error_logger.error(f"Не найден файл с базой данных: {e}")
        sys.exit(1)
    except pd.errors.DatabaseError as e:
        error_logger.error(f"Запрос к несуществующей таблице или атрибуту: {e}")
        sys.exit(1)


def processing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Выполняет обработку данных для подготовки к обучению модели.

    Parameters
    ----------
    df : `pd.DataFrame`
        Исходный датафрейм с данными о бриллиантах, содержащий колонки:
        'carat', 'cut', 'color', 'clarity', 'price'.

    Returns
    -------
    X : `pd.DataFrame`
        Матрица признаков после всех этапов обработки, содержащая колонки:
        'Карат', 'Огранка', 'Цвет', 'Чистота'.
    y : `pd.Series`
        Целевая переменная после обработки.

    Raises
    ------
    ValueError
        Если после очистки данных не осталось строк для обработки.
    """
    try:
        df = rename_columns(df)
        df = clean_data(df)
        if df.empty:
            raise ValueError("После очистки данных не осталось строк для обучения.")
    except ValueError as e:
        error_logger.error(f"Ошибка при обработке данных. {e}")
        sys.exit(1)
    df = change_types(df)
    X, y = split_data(df)
    info_logger.info("Данные успешно обработаны.")
    return X, y


def load_model_and_metric():
    """
    Загружает ранее сохранённую модель и её метрику из реестра.

    Returns
    -------
    model : `lightgbm.sklearn.LGBMRegressor`
        Объект модели, совместимый с интерфейсом scikit-learn, загруженный из файла.
    model_metric : float
        Значение метрики качества (средняя абсолютная ошибка MAE) для загруженной модели.

    Raises
    ------
    FileNotFoundError
        - Если файл настроек модели (model_settings.json) не найден.
        - Если файл самой модели отсутствует в указанном пути реестра.
    KeyError
        Если в файле настроек отсутствуют обязательные ключи: "file_path" или "mae_cv".
    """
    try:
        with open(MODEL_SETTINGS_PATH, encoding="utf-8") as file:
            model_settings = json.load(file)
            model_path = f"{MODEL_REGISTRY_PATH}/{model_settings["file_path"]}"
            model = joblib.load(model_path)
            model_metric = model_settings["mae_cv"]
        info_logger.info("Модель успешно загружена.")
        return model, model_metric
    except FileNotFoundError as e:
        error_logger.error(
            f"Отсутствует файл с настройками модели или сама модель. {e}"
        )
        sys.exit(1)
    except KeyError as e:
        error_logger.error(
            f"В файле настроек отсутствует ключ: {e}. Не удалось загрузить модель."
        )
        sys.exit(1)


def get_new_metric(model, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Вычисляет метрику качества модели с использованием кросс-валидации.

    Функция оценивает модель с помощью кросс-валидации и возвращает
    усредненную по фолдам MAE.

    Parameters
    ----------
    model : `lightgbm.sklearn.LGBMRegressor`
        Модель машинного обучения, совместимая с scikit-learn.
    X : `pd.DataFrame`
        Матрица признаков.
    y : `pd.Series`
        Целевая переменная (стоимость бриллианта).

    Returns
    -------
    `float`
        MAE модели, рассчитанная с помощью кросс-валидации.
    """
    cv_scores = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")
    new_metric = -cv_scores.mean().round(3)
    return new_metric


def dump_model(model, X: pd.DataFrame, y: pd.Series) -> str:
    """
    Обучает модель на переданных данных и сохраняет её в реестр моделей.

    Parameters
    ----------
    X : `pd.DataFrame`
        Матрица признаков.
    y : `pd.Series`
        Целевая переменная (стоимость бриллианта).
    model : `lightgbm.sklearn.LGBMRegressor`
        Модель машинного обучения, совместимая с scikit-learn.

    Returns
    -------
    `str`
        Имя сохранённого файла модели в формате "diamond_ГГГГ-ММ-ДД.model".

    Raises
    ------
    FileNotFoundError
        Если директория реестра моделей (MODEL_REGISTRY_PATH) не существует
    """
    model = model.fit(X, y)
    current_date = date.today()
    model_name = f"diamond_{current_date}.model"
    model_path = f"{MODEL_REGISTRY_PATH}/{model_name}"
    joblib.dump(model, model_path)
    return model_name


def update_model_settings(model, model_name: str, model_metric: float) -> None:
    """
    Обновляет и сохраняет настройки обученной модели в JSON-файле конфигурации.

    Parameters
    ----------
    model : `lightgbm.sklearn.LGBMRegressor`
        Модель машинного обучения, совместимая с scikit-learn.
    model_name : `str`
        Имя файла для сохранения модели.
    model_metric : `float`
        Значение метрики качества модели.

    Raises
    ------
    FileNotFoundError
        Если директория для сохранения файла не существует.
    """
    new_model_settings = {
        "file_path": model_name,
        "mae_cv": model_metric,
        "model_name": model.__class__.__name__,
        "model_params": model.get_params(),
    }
    with open(MODEL_SETTINGS_PATH, "w", encoding="utf-8") as file:
        json.dump(new_model_settings, file, ensure_ascii=False, indent=4)


def evaluate_and_update_model(
    model, X: pd.DataFrame, y: pd.Series, model_metric: float
) -> None:
    """
    Оценивает качество модели и обновляет реестр при улучшении метрики.

    Parameters
    ----------
    model : `lightgbm.sklearn.LGBMRegressor`
        Модель машинного обучения, совместимая с scikit-learn.
    X : pd.DataFrame
        Матрица признаков.
    y : pd.Series
        Целевая переменная (стоимость бриллианта).
    model_metric : float
        Текущее значение метрики качества (MAE) сохранённой модели.
    """
    new_metric = get_new_metric(model, X, y)
    info_logger.info(f"Новые показатели по MAE: {new_metric}, cтарые: {model_metric}.")

    if new_metric < model_metric:
        new_model_name = dump_model(model, X, y)
        info_logger.info("Добавляем новую модель в реестр.")
        update_model_settings(model, new_model_name, new_metric)
        info_logger.info("Успешно обновлен modelsettings.json.")
    else:
        info_logger.info("Модель не добавилась в реестр.")


def main() -> None:
    """
    Точка входа в скрипт.
    """
    df = load_data()
    X, y = processing_data(df)
    model, model_metric = load_model_and_metric()
    evaluate_and_update_model(model, X, y, model_metric)


if __name__ == "__main__":
    main()

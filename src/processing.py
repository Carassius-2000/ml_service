"""
Модуль предоставляет функции для предварительной обработки данных pandas.DataFrame.

Основные возможности:
- Переименование колонок в соответствии с русскоязычными названиями;
- Очистка данных от пропущенных значений и дубликатов;
- Оптимизация типов данных для экономии памяти;
- Разделение данных на признаки и целевую переменную.
"""

import pandas as pd


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переименовывает колонки в соответствии с заданным словарем.

    Осуществляет преобразование английских названий колонок в русские:
    - 'carat' -> 'Карат';
    - 'cut' -> 'Огранка';
    - 'color' -> 'Цвет';
    - 'clarity' -> 'Чистота';
    - 'price' -> 'Цена'.

    Parameters
    ----------
    df : `pd.DataFrame`
        Исходный датафрейм с английскими названиями колонок.

    Returns
    -------
    `pd.DataFrame`
        Новый датафрейм с переименованными колонками.
    """
    columns_mapping = {
        "carat": "Карат",
        "cut": "Огранка",
        "color": "Цвет",
        "clarity": "Чистота",
        "price": "Цена",
    }
    df = df.rename(columns=columns_mapping)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет две операции очистки:
    1. Удаляет все строки, содержащие хотя бы одно пропущенное значение (NaN).
    2. Удаляет полностью дублирующиеся строки, сохраняя только первое вхождение.

    Parameters
    ----------
    df : `pd.DataFrame`
        Исходный датафрейм для очистки.

    Returns
    -------
    `pd.DataFrame`
        Очищенный датафрейм без пропусков и дубликатов.
    """
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def change_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Оптимизирует типы данных для снижения потребления памяти.

    Выполняет следующие преобразования:
    - Все количественные переменные преобразуются в float32.
    - Все категориальные переменные преобразуются в category.

    Parameters
    ----------
    df : `pd.DataFrame`
        Исходный датафрейм с неоптимизированными типами данных.

    Returns
    -------
    `pd.DataFrame`
        Новый датафрейм с оптимизированными типами данных.
    """
    df_new = df.copy()
    for column in df_new.select_dtypes(include=["float64", "int64"]):
        df_new[column] = df_new[column].astype("float32")
    for column in df_new.select_dtypes(include="object"):
        df_new[column] = df_new[column].astype("category")
    return df_new


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет датафрейм на матрицу признаков и целевую переменную.

    Parameters
    ----------
    df : `pd.DataFrame`
        Датафрейм, содержащий все данные включая целевую переменную.

    Returns
    -------
    X : `pd.DataFrame`
        Матрица признаков (все колонки кроме "Цена").
    y : `pd.Series`
        Целевая переменная (значения колонки "Цена").
    """
    y = df["Цена"]
    X = df.drop(columns=["Цена"])
    return X, y

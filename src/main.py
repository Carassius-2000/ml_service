from functools import lru_cache
import json

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

from loggers import error_logger, info_logger
from processing import change_types
from pydantic_models import Diamond

MODEL_REGISTRY_PATH = "model_registry"
MODEL_SETTINGS_PATH = f"{MODEL_REGISTRY_PATH}/modelsettings.json"

app = FastAPI(
    title="ML Web Service fo diamond price prediction",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    """
    Возвращает HTML-ответ с настроенной страницей Swagger UI,
    которая предоставляет интерфейс для взаимодействия с API.

    Используется для настройки кастомного Swagger UI.

    Returns
    -------
    `HTMLResponse`
        HTML-страница с интерфейсом Swagger UI.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect() -> HTMLResponse:
    """
    Возвращает HTML-ответ, используемый при аутентификации OAuth2
    через Swagger UI.

    Returns
    -------
    `HTMLResponse`
        HTML-ответ, необходимый для завершения OAuth2 flow в Swagger UI.
    """
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def custom_redoc() -> HTMLResponse:
    """
    Возвращает HTML-ответ с интерфейсом ReDoc, который предоставляет
    альтернативный способ просмотра документации OpenAPI.
    Используется для настройки кастомного интерфейса ReDoc.

    Returns
    -------
    `HTMLResponse`
        HTML-страница с интерфейсом ReDoc.
    """
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@lru_cache(maxsize=1)
def load_model():
    """
    Загружает модель машинного обучения из реестра.

    Returns
    -------
    model : `lightgbm.sklearn.LGBMRegressor`
        Объект модели, совместимый с интерфейсом scikit-learn, загруженный из файла.

    Raises
    ------
    HTTPException
        - 404 Not Found: Если отсутствует файл настроек модели или сама модель.
        - 422 Unprocessable Entity: Если в JSON-файле настроек отсутствуют обязательные ключи.
    """
    try:
        with open(MODEL_SETTINGS_PATH, encoding="utf-8") as file:
            model_settings = json.load(file)
        model_path = f"{MODEL_REGISTRY_PATH}/{model_settings['file_path']}"
        model = joblib.load(model_path)
        return model
    except FileNotFoundError as e:
        message_error = f"Отсутствует файл с настройками модели или сама модель. {e}"
        error_logger.error(message_error)
        raise HTTPException(
            status_code=404,
            detail=message_error,
        )
    except KeyError as e:
        message_error = (
            f"В файле настроек отсутствует ключ: {e}. Не удалось загрузить модель."
        )
        error_logger.error(message_error)
        raise HTTPException(
            status_code=422,
            detail=message_error,
        )


@app.post("/update_model")
def update_model() -> dict[str, str]:
    """
    Обработчик POST-запроса для принудительного обновления кэшированной модели.

    Очищает LRU-кэш функции загрузки модели, повторно загружает модель из реестра.

    Returns
    -------
    `dict[str, str]`
        Сообщение об успешном обновлении модели.
    """
    load_model.cache_clear()
    model = load_model()
    info_logger.info("Модель успешно обновлена.")
    return {"message": "Модель успешно обновлена."}


@app.post("/diamond_price")
def get_diamond_price_prediction(diamond: Diamond) -> dict[str, float]:
    """
    Обработчик POST-запроса для прогнозирования цены бриллианта.

    Parameters
    ----------
    diamond : `Diamond`
        Pydantic-модель, содержащая характеристики бриллианта для прогнозирования.

    Returns
    -------
    `dict[str, float]`
        Прогноз по стоимости бриллианта.
    """
    df = pd.DataFrame([diamond.model_dump()])
    df = change_types(df)
    model = load_model()
    info_logger.info("Модель успешно загружена.")
    prediction = model.predict(df).round(3)[0]
    info_logger.info("Прогноз по бриллианту успешно получен.")
    return {"price": prediction}


@app.get("/")
async def root() -> dict[str, str]:
    """
    Точка входа в API. Используется для проверки работоспособности сервера.

    Returns
    -------
    `dict[str, str]`
        JSON, который подтверждает, что сервер с API успешно запущен.
    """
    return {"message": "Сервер с API запущен."}

# Описание проекта
Machine Learning Service для [прогнозирования стоимости бриллиантов](https://github.com/Carassius-2000/regression_task). Включает в себя 2 сервиса: `Training` и `Inference`. `Training` используется для обновления модели на основе новых данных. `Inference` используется для получения прогнозов через API.
#### Принцип работы Training
1. Скрипт берет новые данные из SQLite и создает новую версию модели.
2. Новая модель проходит этап валидации по метрике `MAE` перед записью в реестр моделей. Если `MAE` новой модели меньше текущей, то она добавляется в реестр. Также обновляются метаданные в `modelsettings.json`.
3. На этапе развертывания новая версия модели становится доступной в `Inference`.
#### Inference Endpoints 
1. `/`: Домашняя страница Web API. Используется для проверки работоспособности сервиса.
2. `/diamond_price`: Возвращает прогноз по бриллианту и кэширует ML модель.
3. `/update_model`: Обновляет ML модель. Берет ее из `modelsettings.json`.
### Обзор директорий проекта
- `model_registry`: реестр с моделями машинного обучения.
- `src`: исходный код проекта.
- `static`: статика для документации к `Inference` API. Позволяет получать доступ к `OpenAPI`, `ReDocly` без интернета.
- `training`: файлы для `Training`.
### Обзор файлов проекта
- `modelsettings.json`: метаданные ML модели.
- `training.py`: скрипт для обновления модели.
- `diamond.db`: файл c БД в SQLite.
- `loggers.py`: модуль для логирования. В нем описано 2 логера: `info_logger` для stdout и `error_logger` для stderr потоков.
- `processing.py`: модуль для обработки данных. Используется как для `Training`, так и для `Inference`.
- `pydantic_models.py`: Pydantic модель для `Inference`.
- `main.py`: Web API на FastAPI для `Inference`.

## Запуск сервиса Training
1. Клонируйте репозиторий. 
```bash 
git clone https://github.com/Carassius-2000/ml_service
```

2. Установите Docker.

3. Соберите и запустите сервис.
```bash
docker compose --profile training up --build
```

## Запуск сервиса API
1. Клонируйте репозиторий. 
```bash 
git clone https://github.com/Carassius-2000/ml_service
```

2. Установите Docker.

3. Соберите и запустите сервис.
```bash
docker compose --profile api up --build
```
4. Проверить работоспособность системы можно либо открыв в браузере [http://localhost:81/](http://localhost:81/), либо посмотрев состояние контейнеров. Должен быть запущен контейнер `api`.
```bash
docker compose ps -a
```
С документацией Web API можно ознакомиться двумя способами:
- Swagger/OpenAPI: [http://localhost:81//docs](http://localhost:81//docs).
- Redocly: [http://localhost:81//redoc](http://localhost:81//redoc).

Для того чтобы остановить работу контейнера выполните:
```bash
docker compose stop
```
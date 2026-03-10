# UI приложение (Linux/macOS)

Отдельное веб-UI (Flask) для загрузки изображения и запуска распознавания через API.

## Запуск одной командой (API + UI)

```bash
docker compose up --build
```

После старта:
- API: `http://localhost:8000/health`
- UI: `http://localhost:7860`

## Что в UI
- загрузка изображения,
- кнопка "Распознать",
- разметка OCR-боксов на изображении,
- полный OCR-текст,
- JSON с полями,
- сохранённый файл `ui_outputs/result.json`.

## Если нужен только API

```bash
docker compose up --build doc-api
```

## Параметры UI

- `OCR_API_URL` — адрес backend endpoint, по умолчанию `http://doc-api:8000/process`.
- `UI_OUTPUT_DIR` — папка сохранения `result.json`, по умолчанию `ui_outputs`.

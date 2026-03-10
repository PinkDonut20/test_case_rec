# UI приложение (Linux/macOS)

Отдельное UI для загрузки изображения и запуска распознавания через API.

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
- отрисованные OCR-боксы,
- полный OCR-текст,
- JSON с полями,
- сохранённый файл `ui_outputs/result.json`.

## Если нужен только API

```bash
docker compose up --build doc-api
```

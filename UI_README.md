# UI приложение (Linux/macOS)

Отдельное простое UI для вашего API.

## Что делает
- Загружает изображение документа.
- Отправляет его в `POST /process` вашего backend API.
- Показывает OCR-боксы поверх изображения.
- Показывает OCR-текст и извлечённые поля.
- Сохраняет полный JSON в `ui_outputs/result.json`.

## Запуск

1) Поднимите backend API (как раньше):

```bash
docker compose up --build
```

2) В отдельном терминале запустите UI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ui_requirements.txt
python ui_app.py
```

3) Откройте в браузере:

```text
http://localhost:7860
```

## Настройки

- `OCR_API_URL` — URL backend `/process` (по умолчанию `http://localhost:8000/process`).
- `UI_OUTPUT_DIR` — куда сохранять JSON UI (по умолчанию `ui_outputs`).

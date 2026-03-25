# UI приложение (Linux/macOS) — Doc OCR Studio

Отдельный красивый UI-клиент для вашего API (`ui_app.py`).

## Что добавлено
- современный визуальный стиль (градиент, карточки, аккуратные блоки),
- вкладки: **Поля / OCR текст / Полный JSON**,
- статус выполнения с понятными сообщениями,
- сохранение JSON с timestamp в `ui_outputs/` + скачивание файла из интерфейса.

## Что делает
- Загружает изображение документа.
- Отправляет его в `POST /process` backend API.
- Рисует OCR-боксы поверх изображения.
- Показывает OCR-текст и извлечённые поля.

## Запуск

1) Поднимите backend API:

```bash
docker compose up --build
```

2) В другом терминале запустите UI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ui_requirements.txt
python ui_app.py
```

3) Откройте:

```text
http://localhost:7860
```

## Настройки
- `OCR_API_URL` — URL backend `/process` (по умолчанию `http://localhost:8000/process`).
- `UI_OUTPUT_DIR` — директория для JSON UI (по умолчанию `ui_outputs`).

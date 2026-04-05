# UI приложение (Linux/macOS) — Desktop (Tkinter)

Полностью переписанный UI без Gradio, чтобы запускался стабильно в обычном Python-окружении.

## Почему теперь точно проще
- `tkinter` встроен в Python (обычно уже есть в macOS/Linux установке),
- нет конфликтов `gradio` / `huggingface_hub`,
- минимум зависимостей: только `requests` и `Pillow`.

## Что умеет UI
- Загрузить изображение документа.
- Отправить в API (`POST /process`).
- Показать OCR-боксы на изображении.
- Показать `fields`, OCR-текст и полный JSON (вкладки).
- Автосохранение JSON в `ui_outputs/` + ручное `Сохранить JSON`.

## Запуск

1) Поднимите backend API:

```bash
docker compose up --build
```

2) В другом терминале запустите desktop UI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ui_requirements.txt
python ui_app.py
```

## Настройки
- `OCR_API_URL` — URL backend `/process` (по умолчанию `http://localhost:8000/process`).
- `UI_OUTPUT_DIR` — директория для JSON UI (по умолчанию `ui_outputs`).

## Если `tkinter` не найден
На некоторых Linux-сборках может не быть пакета tk:

```bash
sudo apt-get update && sudo apt-get install -y python3-tk
```

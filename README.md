# Финальная версия (супер просто): OCR строки → Qwen → JSON

Сделал максимально просто под тестовое:
1. Берем изображение
2. OCR собирает весь текст по строкам (рус+англ)
3. Отдаем строки + общий текст в LLM (Qwen)
4. Возвращаем JSON с главными полями

## Что возвращает API

`POST /process` вернет:
- `ocr.lines` — все строки OCR с bbox и confidence
- `ocr.full_text` — весь OCR-текст одной строковой массой
- `fields` — главная инфа из LLM:
  - `full_name`
  - `birth_date`
  - `document_number`
- `llm.error` — текст ошибки, если LLM не смогла вернуть JSON

## Быстрый запуск на MacBook

### 1) Поднять Ollama + модель

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

### 2) Запуск API

```bash
docker compose up --build
```

### 3) Проверка

```bash
curl http://localhost:8000/health
```

### 4) Обработка файла

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## ENV

В `.env`:
- `LLM_BASE_URL` (по умолчанию `http://host.docker.internal:11434/v1`)
- `LLM_API_KEY` (для Ollama можно `ollama`)
- `LLM_MODEL` (по умолчанию `qwen2.5:7b-instruct`)

## Почему это ок для задания

- Логика прозрачная: OCR -> LLM -> JSON
- Без переусложнения
- Быстро поднимается в Docker
- На макбуке работает CPU-only

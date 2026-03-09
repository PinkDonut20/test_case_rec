# Финальная версия: OCR → LLM → JSON (API + Docker)

Сделал **максимально простой и быстрый** pipeline под твой сценарий:
1. OCR собирает текст **по строкам**
2. строки уходят в LLM
3. LLM возвращает JSON с ключевыми полями

## Что внутри

- `app.py` — FastAPI сервис
- `Dockerfile` + `docker-compose.yml` — запуск одной командой
- `requirements.txt`
- `outputs/` — артефакты

## Логика

`/process`:
- принимает картинку
- выравнивает документ (OpenCV)
- распознает строки OCR (Tesseract `rus+eng`)
- отправляет список строк в LLM
- получает и возвращает JSON:
  - `full_name`
  - `birth_date`
  - `document_number`
- сохраняет:
  - выровненное изображение
  - изображение с line-box
  - полный `*_result.json`

## Почему так (быстро и под MacBook)

- OCR через Tesseract — лёгкий CPU вариант
- LLM через OpenAI-compatible endpoint (`LLM_BASE_URL`), по умолчанию под Ollama на Mac:
  - `http://host.docker.internal:11434/v1`
- Никаких сложных GPU зависимостей в контейнере

## Быстрый старт (MacBook)

### 1) Подними Ollama и модель

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

### 2) Запусти API

```bash
docker compose up --build
```

API будет на `http://localhost:8000`.

### 3) Проверка

```bash
curl http://localhost:8000/health
```

### 4) Обработка изображения

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## ENV

Скопируй `.env.example` в `.env` при необходимости:

- `LLM_BASE_URL` — URL LLM API
- `LLM_API_KEY` — ключ (для Ollama можно `ollama`)
- `LLM_MODEL` — модель (`qwen2.5:7b-instruct` по умолчанию)

## Важно

- По требованию: LLM используется как основной extractor.
- Если LLM вернет невалидный ответ, ошибка кладется в `llm.error`, поля будут `null`.

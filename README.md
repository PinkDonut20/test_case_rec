# OCR API для тестового: запускается без Ollama

Сделал так, чтобы **просто поднять Docker и сразу получить распознавание**.

## Ключевая идея

- По умолчанию сервис работает в режиме `heuristic`:
  - OCR собирает весь текст по строкам
  - локально извлекаются `full_name`, `birth_date`, `document_number`
  - **никакой Ollama запускать не нужно**

- Если нужно, можно включить режим `api`:
  - OCR строки + full text отправляются во внешний OpenAI-compatible LLM
  - например в Qwen/Ollama endpoint

## Быстрый старт (без Ollama)

```bash
docker compose up --build
```

Проверка:

```bash
curl http://localhost:8000/health
```

Обработка документа:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## Что вернет /process

- `ocr.lines` — строки OCR + bbox + confidence
- `ocr.full_text` — весь OCR-текст
- `fields`:
  - `full_name`
  - `birth_date`
  - `document_number`
- `extractor.mode`:
  - `heuristic` (без LLM)
  - `api` (через внешний LLM)
  - `heuristic_fallback` (если в `api` режиме LLM упал)

## Если захочешь подключить Qwen через API

В `.env`:

```env
EXTRACTOR_MODE=api
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b-instruct
```

## Почему это удобно для задания

- Один `docker compose up --build` и сервис уже рабочий
- Нет обязательной внешней зависимости на Ollama
- При этом можно включить LLM-режим без переписывания кода

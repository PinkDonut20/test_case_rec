# Document OCR Pipeline (тестовое задание, финальная версия)

## Что это

Это рабочий baseline-сервис для первичной обработки персональных документов:

1. принимает изображение документа,
2. распознаёт текст,
3. извлекает структурированные поля,
4. возвращает JSON и сохраняет артефакты обработки.

Проект сделан как self-contained репозиторий для запуска через Docker Compose.

---

## Архитектура

### 1) OCR слой
- Используется **EasyOCR** (`ru`, `en`) для локального инференса.
- Для повышения устойчивости применяется multi-pass OCR:
  - original,
  - grayscale,
  - CLAHE,
  - sharpen,
  - upscale x1.5.
- Затем выполняется:
  - дедупликация похожих боксов,
  - агрегация в строки,
  - фильтрация шумных детекций.

**Почему так:** на документах с защитным фоном и слабым контрастом один проход часто пропускает важные токены; multi-pass заметно поднимает recall.

### 2) Извлечение полей
Поддерживаются три режима (`EXTRACTOR_MODE`):

- `heuristic` — только локальные правила,
- `hf` — локальная модель HuggingFace (по умолчанию),
- `api` — внешний OpenAI-compatible endpoint.

В любом режиме есть страховка:
- пост-обработка результата (`full_name`, `birth_date`, `document_number`),
- fallback на эвристики при ошибках модели.

**Почему так:** это даёт баланс между стабильностью офлайн-режима и качеством финализации через модель.

### 3) API слой
FastAPI:
- `GET /health` — статус и активная конфигурация,
- `POST /process` — полный пайплайн обработки.

Сервис сохраняет:
- `outputs/*_annotated.jpg`,
- `outputs/*_result.json`.

---

## Выбранные модели и обоснование

### OCR: EasyOCR
- лучше, чем pytesseract, держит сложные фоны документов,
- работает локально,
- поддерживает русский/английский текст,
- подходит под CPU и GPU режим.

### Финализация полей (по умолчанию): HF модель
- по умолчанию: `Qwen/Qwen2.5-0.5B-Instruct`,
- достаточно лёгкая для быстрого старта,
- умеет аккуратно структурировать OCR-текст в JSON.

### Альтернатива: внешний API
- поддержан OpenAI-compatible путь (`EXTRACTOR_MODE=api`),
- удобно подключать внешний endpoint при необходимости.

---

## Быстрый запуск

```bash
docker compose up --build
```

После запуска доступны:
- API: `http://localhost:8000`
- UI: `http://localhost:7860`

Проверка API:

```bash
curl http://localhost:8000/health
```

Запуск обработки:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

---

## Конфигурация (`.env`)

Пример:

```env
EXTRACTOR_MODE=hf
USE_GPU=0
OCR_CONF_THRESHOLD=0.20
OCR_MIN_BOX_AREA=60
HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# только для EXTRACTOR_MODE=api
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b-instruct
```

---



## Формат ответа `/process`

- `ocr.lines` — OCR-строки с bbox и confidence,
- `ocr.full_text` — полный OCR текст,
- `ocr.lines_count` — число строк после фильтрации,
- `fields`:
  - `full_name`,
  - `birth_date`,
  - `document_number`,
- `extractor` — какой режим использовался и возможная ошибка.

---

## Совместимость с окружением проверки

Проект подготовлен под запуск в Linux + Docker Compose и учитывает CPU/GPU сценарии.
Рекомендованный способ проверки — `docker compose up --build` и запрос на `POST /process`.


## UI (для ручного теста)

Откройте `http://localhost:7860`, загрузите изображение, нажмите **Распознать**.
UI покажет OCR-боксы, текст и сохранит `ui_outputs/result.json`.

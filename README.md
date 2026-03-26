# Document OCR Pipeline (refactored)

Проект переработан с учетом фидбека:
- код разбит на модули,
- добавлен современный OCR backend (`LightOnOCR-2-1B`) + fallback,
- structured extraction вынесен в отдельный слой, API-first.

## Что делает сервис
1. Принимает изображение документа (`POST /process`).
2. Запускает OCR (по умолчанию `lighton`, fallback `easyocr`).
3. Извлекает `full_name`, `birth_date`, `document_number`:
   - `api` (внешний LLM, primary),
   - `hf` (локальный fallback),
   - `hybrid` (сначала API, затем HF, затем эвристики),
   - `heuristic` (только локальные правила).
4. Возвращает JSON и сохраняет `outputs/*_annotated.jpg`, `outputs/*_result.json`.

## Новая структура

```text
app.py
src/
  config.py
  ocr.py
  extraction.py
  pipeline.py
```

- `src/ocr.py` — OCR backend'ы и fallback-логика.
- `src/extraction.py` — API/HF/heuristic extraction + postprocessing.
- `src/pipeline.py` — orchestration, сохранение артефактов.
- `app.py` — только HTTP слой.

## Почему такие модели

### OCR
- **Primary**: `lightonai/LightOnOCR-2-1B` (через `transformers` image-text-to-text).
- **Fallback**: `easyocr` для устойчивости при проблемах загрузки/инференса primary backend.

### Structured extraction
- По умолчанию используется локальный `hf` режим, который не требует внешних ключей.
- API-режим оставлен только как опциональный сценарий (если хотите подключить внешний endpoint вручную).

## Быстрый запуск

```bash
cp .env.example .env
docker compose up --build
# запускается без LLM ключей
```

Проверка:

```bash
curl http://localhost:8000/health
```

Запрос обработки:

```bash
curl -X POST "http://localhost:8000/process" -F "file=@/path/to/document.jpg"
```

## Конфигурация

```env
# hf | heuristic | hybrid | api
EXTRACTOR_MODE=hf

# lighton | easyocr
OCR_BACKEND=lighton
USE_GPU=0
OCR_CONF_THRESHOLD=0.20
OCR_MIN_BOX_AREA=60
LIGHTON_MODEL=lightonai/LightOnOCR-2-1B
HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# Никакие LLM_BASE_URL / LLM_API_KEY для обычного запуска не нужны.
# API режим опционален.
```

## Что улучшать дальше (dewarping)
1. Документный детектор (quad/segmentation) перед OCR.
2. Геометрическая коррекция (perspective + local warping).
3. OCR-quality scorer (blur/glare/skew) и автоматический повтор с другими препроцессами.
4. Тестовый набор + метрики CER/WER + extraction accuracy.

# Финальная версия сервиса: OCR + финализация полей

## Что важно

- Текст извлекается через EasyOCR multi-pass.
- Финализация полей (`full_name`, `birth_date`, `document_number`) выполняется моделью по всем OCR-строкам и полному тексту:
  - по умолчанию локальная модель HF в режиме `text-generation` (`EXTRACTOR_MODE=hf`),
  - альтернативно внешний endpoint (`EXTRACTOR_MODE=api`).
- При ошибке модели включается безопасный fallback на локальные эвристики.

## Важный момент по HF

Для режима `hf` нужна **instruction-модель**, которая умеет `text-generation`.
Рекомендуемый дефолт:
- `HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct`

Если поставить seq2seq-модель (например flan) в этот режим, можно получить не-JSON ответ.

## Режимы

- `heuristic` — только локальные правила
- `hf` — локальная HuggingFace-модель (рекомендуемый дефолт)
- `api` — внешний OpenAI-compatible endpoint

## Быстрый запуск

```bash
docker compose up --build
```

Проверка:

```bash
curl http://localhost:8000/health
```

Обработка:

```bash
curl -X POST "http://localhost:8000/process"   -F "file=@/path/to/document.jpg"
```

## Конфигурация (`.env`)

```env
EXTRACTOR_MODE=hf
USE_GPU=0
OCR_CONF_THRESHOLD=0.20
OCR_MIN_BOX_AREA=60
HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct

LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b-instruct
```

## Ответ API

- `ocr.lines` / `ocr.full_text` / `ocr.lines_count`
- `fields.full_name`, `fields.birth_date`, `fields.document_number`
- `extractor.mode`, `extractor.error`

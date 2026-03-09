# Финальный OCR API: EasyOCR multi-pass + улучшенный парсинг полей

Сделал финальный вариант, чтобы лучше детектить текст на сложных документах.

## Что улучшено

- OCR теперь **multi-pass**:
  - original
  - gray
  - CLAHE
  - sharpen
  - upscale x1.5
- Результаты из проходов объединяются (дедуп + слияние), потом склеиваются в строки.
- Мусор режется на этапе OCR:
  - `OCR_CONF_THRESHOLD` (по умолчанию `0.20`)
  - `OCR_MIN_BOX_AREA` (по умолчанию `60`)
- Парсинг полей стал точнее:
  - ФИО сначала ищется по якорям (`Фамилия`, `Имя`, `Отчество`), затем fallback
  - дата рождения ищется возле `ДАТА РОЖД...`/`BIRTH`
  - номер документа по приоритетным паттернам

## Запуск

```bash
docker compose up --build
```

Проверка:

```bash
curl http://localhost:8000/health
```

Обработка:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## Режимы extractor

- `EXTRACTOR_MODE=heuristic` (по умолчанию)
- `EXTRACTOR_MODE=api` — отправка OCR в OpenAI-compatible LLM

## ENV

```env
EXTRACTOR_MODE=heuristic
USE_GPU=0
OCR_CONF_THRESHOLD=0.20
OCR_MIN_BOX_AREA=60

# только для EXTRACTOR_MODE=api
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b-instruct
```

## Ответ API

- `ocr.engine = easyocr_multipass`
- `ocr.lines`, `ocr.full_text`, `ocr.lines_count`
- `fields.full_name`, `fields.birth_date`, `fields.document_number`
- `extractor.mode`, `extractor.error`

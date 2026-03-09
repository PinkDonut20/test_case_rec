# OCR API для тестового: EasyOCR вместо pytesseract

По твоему комменту переписал OCR-движок:
- было: `pytesseract`
- стало: **`EasyOCR`** (лучше на документах с шумом/фоном)

## Что сейчас делает сервис

1. Принимает изображение (`POST /process`)
2. OCR через EasyOCR (ru+en)
3. Фильтрует слабые/мусорные боксы
4. Извлекает поля:
   - `heuristic` (по умолчанию, локально)
   - или `api` (через OpenAI-compatible LLM)
5. Возвращает JSON + сохраняет annotated image и result json

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
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## Режимы

- `EXTRACTOR_MODE=heuristic` (default)
- `EXTRACTOR_MODE=api` + `LLM_BASE_URL`/`LLM_MODEL` для Qwen/Ollama

## ENV

Смотри `.env.example`:
- `EXTRACTOR_MODE`
- `USE_GPU` (0/1)
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

## Что в ответе

- `ocr.engine = easyocr`
- `ocr.lines` / `ocr.full_text` / `ocr.lines_count`
- `fields.full_name`, `fields.birth_date`, `fields.document_number`
- `extractor.mode` и `extractor.error`

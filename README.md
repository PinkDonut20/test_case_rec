# OCR API для тестового: лучше фильтрация шума + точнее поля

Сервис запускается сразу через Docker и по умолчанию **без Ollama**.

## Что улучшено

- OCR теперь идет через предобработку изображения (denoise + adaptive threshold), чтобы меньше ловить шум/фон.
- Добавлена фильтрация мусора:
  - минимальный confidence слова
  - отбрасывание слишком маленьких боксов
  - отбрасывание «непохожих на текст» токенов
- Эвристики полей стали точнее:
  - дата рождения ищется с учетом якорей (`ДАТА РОЖД...` / `BIRTH`)
  - номер документа приоритетно формата `XX XX XXXXXX`
  - ФИО фильтруется от служебных слов (`МВД`, `код`, `дата`, `республика` и т.п.)

## Режимы

- `EXTRACTOR_MODE=heuristic` (по умолчанию): OCR + локальное извлечение полей
- `EXTRACTOR_MODE=api`: OCR + внешний OpenAI-compatible LLM
  - если LLM упал/вернул мусор, есть fallback на heuristic

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
- `ocr.lines_count` — сколько строк осталось после фильтрации
- `fields`:
  - `full_name`
  - `birth_date`
  - `document_number`
- `extractor.mode`:
  - `heuristic`
  - `api`
  - `heuristic_fallback`

## Опционально: Qwen через API

В `.env`:

```env
EXTRACTOR_MODE=api
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b-instruct
```

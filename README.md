# Сервис первичной обработки документов (демо-версия)

Проект подготовлен для демонстрации базового production-пайплайна:

1. Принимаем изображение документа.
2. Находим и распознаём текст (EasyOCR, multi-pass).
3. Извлекаем ключевые поля.
4. Возвращаем JSON и сохраняем артефакты обработки.

## Что реализовано

- REST API на FastAPI.
- OCR на EasyOCR (`ru`, `en`) с несколькими предобработками изображения:
  - исходное изображение
  - grayscale
  - CLAHE
  - sharpen
  - upscale x1.5
- Объединение и очистка OCR-детекций:
  - дедупликация близких боксов
  - сборка слов в строки
  - фильтрация слабых и шумных детекций
- Извлечение полей:
  - локальный режим (`heuristic`)
  - внешний режим (`api`) через OpenAI-compatible endpoint
  - автоматический fallback на `heuristic`, если внешний сервис недоступен

## Выходные данные

`POST /process` возвращает:

- `ocr.lines` — строки OCR с координатами и confidence
- `ocr.full_text` — полный текст
- `ocr.lines_count` — количество строк после фильтрации
- `fields`:
  - `full_name`
  - `birth_date`
  - `document_number`
- `extractor` — техническая информация о режиме извлечения

Также в `outputs/` сохраняются:

- `*_annotated.jpg`
- `*_result.json`

## Локальный запуск

```bash
docker compose up --build
```

Проверка статуса:

```bash
curl http://localhost:8000/health
```

Пример обработки:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/document.jpg"
```

## Конфигурация

Параметры задаются через `.env`:

```env
EXTRACTOR_MODE=heuristic
USE_GPU=0
OCR_CONF_THRESHOLD=0.20
OCR_MIN_BOX_AREA=60

# требуется только для EXTRACTOR_MODE=api
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b-instruct
```

## Примечание для показа

Для демонстрации без внешних зависимостей рекомендуется режим `heuristic`.
Он запускается сразу после `docker compose up --build` и подходит для офлайн-прогона сценария end-to-end.

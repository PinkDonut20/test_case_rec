import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI(title="doc-ocr-llm", version="2.1.0")

OUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# По умолчанию НЕ требуем Ollama/внешний LLM.
# Режимы:
# - heuristic (default): OCR + локальное извлечение полей
# - api: OCR + внешний OpenAI-compatible LLM
EXTRACTOR_MODE = os.getenv("EXTRACTOR_MODE", "heuristic").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")


def ocr_lines(image_bgr) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb,
        lang="rus+eng",
        output_type=pytesseract.Output.DICT,
        config="--oem 1 --psm 6",
    )

    lines_map: dict[tuple[int, int, int], dict[str, Any]] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]) != "-1" else -1.0
        if not text or conf < 20:
            continue

        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        if key not in lines_map:
            lines_map[key] = {
                "words": [],
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf_sum": 0.0,
                "cnt": 0,
            }
        item = lines_map[key]
        item["words"].append(text)
        item["x"] = min(item["x"], int(data["left"][i]))
        item["y"] = min(item["y"], int(data["top"][i]))
        item["w"] = max(item["w"], int(data["left"][i]) + int(data["width"][i]) - item["x"])
        item["h"] = max(item["h"], int(data["top"][i]) + int(data["height"][i]) - item["y"])
        item["conf_sum"] += conf
        item["cnt"] += 1

    lines = []
    for _, v in sorted(lines_map.items(), key=lambda kv: (kv[1]["y"], kv[1]["x"])):
        lines.append(
            {
                "text": re.sub(r"\s+", " ", " ".join(v["words"]).strip()),
                "bbox": [v["x"], v["y"], v["w"], v["h"]],
                "confidence": round(v["conf_sum"] / max(v["cnt"], 1), 2),
            }
        )
    return lines


def draw_lines(image_bgr, lines: list[dict[str, Any]]):
    out = image_bgr.copy()
    for line in lines:
        x, y, w, h = line["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def heuristic_extract(ocr_lines_text: list[str]) -> dict[str, Any]:
    joined = " ".join(ocr_lines_text)

    birth_date = None
    date_match = re.search(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b", joined)
    if date_match:
        birth_date = date_match.group(1).replace("-", ".").replace("/", ".")

    document_number = None
    num_match = re.search(r"\b(\d{2}\s?\d{2}\s?\d{6}|\d{9,12})\b", joined)
    if num_match:
        document_number = re.sub(r"\s+", " ", num_match.group(1)).strip()

    full_name = None
    stop_words = {"ВОДИТЕЛЬСКОЕ", "УДОСТОВЕРЕНИЕ", "ПАСПОРТ", "РЕСПУБЛИКА", "ФЕДЕРАЦИЯ"}
    for line in ocr_lines_text:
        up = re.sub(r"[^А-ЯЁ\s-]", "", line.upper()).strip()
        words = up.split()
        if len(words) < 2 or len(words) > 4:
            continue
        if any(w in stop_words for w in words):
            continue
        if all(re.fullmatch(r"[А-ЯЁ-]+", w or "") for w in words):
            full_name = up
            break

    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "document_number": document_number,
    }


def llm_extract(ocr_lines_text: list[str], full_text: str) -> dict[str, Any]:
    if not LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL is empty")

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "dummy")
    prompt = (
        "Ниже OCR русского документа. Извлеки главную информацию и верни ТОЛЬКО JSON с ключами: "
        "full_name, birth_date, document_number. Формат даты DD.MM.YYYY, если поле не найдено — null.\n"
        f"OCR lines: {json.dumps(ocr_lines_text, ensure_ascii=False)}\n"
        f"OCR full_text: {full_text}"
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Ответ только JSON без markdown и пояснений."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content or ""
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError(f"LLM did not return JSON: {content[:300]}")

    parsed = json.loads(match.group(0))
    return {
        "full_name": parsed.get("full_name"),
        "birth_date": parsed.get("birth_date"),
        "document_number": parsed.get("document_number"),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "extractor_mode": EXTRACTOR_MODE,
        "llm_model": LLM_MODEL if EXTRACTOR_MODE == "api" else None,
        "llm_base_url": LLM_BASE_URL if EXTRACTOR_MODE == "api" else None,
    }


@app.post("/process")
async def process(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "input.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    image = cv2.imread(str(tmp_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    lines = ocr_lines(image)
    line_texts = [x["text"] for x in lines]
    full_text = "\n".join(line_texts)

    try:
        if EXTRACTOR_MODE == "api":
            fields = llm_extract(line_texts, full_text)
            used = "api"
        else:
            fields = heuristic_extract(line_texts)
            used = "heuristic"
        extractor_error = None
    except Exception as e:
        fields = heuristic_extract(line_texts)
        used = "heuristic_fallback"
        extractor_error = str(e)

    stem = Path(file.filename or "input").stem
    annotated_path = OUT_DIR / f"{stem}_annotated.jpg"
    json_path = OUT_DIR / f"{stem}_result.json"
    cv2.imwrite(str(annotated_path), draw_lines(image, lines))

    payload = {
        "input_image": file.filename,
        "annotated_image": str(annotated_path),
        "ocr": {"lines": lines, "full_text": full_text},
        "fields": fields,
        "extractor": {
            "mode": used,
            "error": extractor_error,
            "llm_model": LLM_MODEL if used.startswith("api") else None,
            "llm_base_url": LLM_BASE_URL if used.startswith("api") else None,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return JSONResponse(payload)

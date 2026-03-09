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

app = FastAPI(title="doc-ocr-llm", version="2.0.0")

OUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
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


def extract_with_llm(ocr_lines_text: list[str], full_text: str) -> dict[str, Any]:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    prompt = (
        "Ниже OCR русского документа. "
        "Извлеки главную информацию и верни ТОЛЬКО JSON с ключами: "
        "full_name, birth_date, document_number. "
        "Формат даты DD.MM.YYYY, если поле не найдено — null.\n"
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
    return {"status": "ok", "llm_model": LLM_MODEL, "llm_base_url": LLM_BASE_URL}


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

    # супер-просто: OCR по исходному изображению, без сложной доп. логики
    lines = ocr_lines(image)
    line_texts = [x["text"] for x in lines]
    full_text = "\n".join(line_texts)

    try:
        fields = extract_with_llm(line_texts, full_text)
        llm_error = None
    except Exception as e:
        fields = {"full_name": None, "birth_date": None, "document_number": None}
        llm_error = str(e)

    stem = Path(file.filename or "input").stem
    annotated_path = OUT_DIR / f"{stem}_annotated.jpg"
    json_path = OUT_DIR / f"{stem}_result.json"

    cv2.imwrite(str(annotated_path), draw_lines(image, lines))

    payload = {
        "input_image": file.filename,
        "annotated_image": str(annotated_path),
        "ocr": {
            "lines": lines,
            "full_text": full_text,
        },
        "fields": fields,
        "llm": {
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "error": llm_error,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return JSONResponse(payload)

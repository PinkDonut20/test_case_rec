import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI(title="doc-ocr-llm", version="3.0.0")

OUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTOR_MODE = os.getenv("EXTRACTOR_MODE", "heuristic").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
USE_GPU = os.getenv("USE_GPU", "0") == "1"

_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ru", "en"], gpu=USE_GPU)
    return _reader


def preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return thr


def _is_text_like(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return sum(ch.isalnum() for ch in s) >= 2


def ocr_lines(image_bgr) -> list[dict[str, Any]]:
    prepared = preprocess_for_ocr(image_bgr)
    reader = get_reader()
    result = reader.readtext(prepared, detail=1, paragraph=False)

    lines: list[dict[str, Any]] = []
    h_img, w_img = prepared.shape[:2]
    min_box_area = max(120, int(h_img * w_img * 0.0003))

    for item in result:
        box, text, conf = item
        text = re.sub(r"\s+", " ", str(text).strip())
        conf = float(conf)
        if conf < 0.35 or not _is_text_like(text):
            continue

        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        if w * h < min_box_area:
            continue

        lines.append(
            {
                "text": text,
                "bbox": [x, y, w, h],
                "confidence": round(conf, 3),
            }
        )

    lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return lines


def draw_lines(image_bgr, lines: list[dict[str, Any]]):
    out = image_bgr.copy()
    for line in lines:
        x, y, w, h = line["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def _parse_date(s: str):
    m = re.search(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b", s)
    if not m:
        return None
    value = m.group(1).replace("-", ".").replace("/", ".")
    try:
        dt = datetime.strptime(value, "%d.%m.%Y")
    except ValueError:
        return None
    if dt.year < 1930 or dt.year > datetime.now().year + 1:
        return None
    return value


def heuristic_extract(ocr_lines_text: list[str]) -> dict[str, Any]:
    lines = [x for x in (_normalize(t) for t in ocr_lines_text) if x]
    joined = " ".join(lines)

    birth_date = None
    anchor_idx = None
    for i, line in enumerate(lines):
        if "ДАТА РОЖД" in line or "BIRTH" in line:
            anchor_idx = i
            break
    if anchor_idx is not None:
        window = lines[max(0, anchor_idx - 1): min(len(lines), anchor_idx + 3)]
        for line in window:
            birth_date = _parse_date(line)
            if birth_date:
                break
    if not birth_date:
        birth_date = _parse_date(joined)

    doc_patterns = [r"\b(\d{2}\s?\d{2}\s?\d{6})\b", r"\b(\d{9,12})\b"]
    document_number = None
    for p in doc_patterns:
        m = re.search(p, joined)
        if m:
            document_number = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    stop_words = {
        "ВОДИТЕЛЬСКОЕ", "УДОСТОВЕРЕНИЕ", "ПАСПОРТ", "РЕСПУБЛИКА", "ФЕДЕРАЦИЯ", "РОССИЙСКАЯ",
        "ДАТА", "ВЫДАЧИ", "КОД", "ПОДРАЗДЕЛЕНИЯ", "МВД", "ПО", "ГОРОД", "ГОР",
    }
    full_name = None
    for line in lines:
        clean = re.sub(r"[^А-ЯЁ\s-]", "", line).strip()
        words = [w for w in clean.split() if w]
        if not (2 <= len(words) <= 4):
            continue
        if any(w in stop_words for w in words):
            continue
        if all(re.fullmatch(r"[А-ЯЁ-]{2,}", w or "") for w in words):
            full_name = " ".join(words)
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
        "Ниже OCR русского документа после фильтрации шума. "
        "Игнорируй служебные строки (МВД, код подразделения, заголовки). "
        "Верни ТОЛЬКО JSON с ключами: full_name, birth_date, document_number. "
        "Дата: DD.MM.YYYY, если нет — null.\n"
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
        "ocr_engine": "easyocr",
        "use_gpu": USE_GPU,
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
        "ocr": {
            "engine": "easyocr",
            "lines": lines,
            "full_text": full_text,
            "lines_count": len(lines),
        },
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

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

app = FastAPI(title="doc-ocr-llm", version="4.0.0")

OUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTOR_MODE = os.getenv("EXTRACTOR_MODE", "heuristic").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
USE_GPU = os.getenv("USE_GPU", "0") == "1"

OCR_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", "0.20"))
OCR_MIN_BOX_AREA = int(os.getenv("OCR_MIN_BOX_AREA", "60"))

_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ru", "en"], gpu=USE_GPU)
    return _reader


def _is_text_like(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    alnum = sum(ch.isalnum() for ch in s)
    return alnum >= 2


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _line_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def build_ocr_variants(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    sharpen_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharp = cv2.filter2D(image_bgr, -1, sharpen_kernel)

    up = cv2.resize(image_bgr, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)

    return [
        ("orig", image_bgr, 1.0),
        ("gray", gray, 1.0),
        ("clahe", clahe, 1.0),
        ("sharp", sharp, 1.0),
        ("up", up, 1.5),
    ]


def _bbox_from_quad(box):
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return x, y, w, h


def _boxes_close(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    dx, dy = abs(acx - bcx), abs(acy - bcy)
    h_ref = max(ah, bh, 1)
    w_ref = max(aw, bw, 1)
    return dx < 0.6 * w_ref and dy < 0.6 * h_ref


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for c in candidates:
        added = False
        for cl in clusters:
            if _boxes_close(c["bbox"], cl[0]["bbox"]):
                cl.append(c)
                added = True
                break
        if not added:
            clusters.append([c])

    merged = []
    for cl in clusters:
        cl_sorted = sorted(cl, key=lambda x: (x["confidence"], len(x["text"])), reverse=True)
        best = cl_sorted[0]
        x1 = min(x["bbox"][0] for x in cl)
        y1 = min(x["bbox"][1] for x in cl)
        x2 = max(x["bbox"][0] + x["bbox"][2] for x in cl)
        y2 = max(x["bbox"][1] + x["bbox"][3] for x in cl)
        merged.append(
            {
                "text": best["text"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": round(sum(x["confidence"] for x in cl) / len(cl), 3),
            }
        )
    return merged


def _merge_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []
    words = sorted(words, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    line_clusters: list[list[dict[str, Any]]] = []
    heights = [max(1, w["bbox"][3]) for w in words]
    median_h = sorted(heights)[len(heights) // 2]
    y_thr = max(8, int(0.6 * median_h))

    for w in words:
        y_mid = w["bbox"][1] + w["bbox"][3] / 2
        placed = False
        for cl in line_clusters:
            cy = sum(x["bbox"][1] + x["bbox"][3] / 2 for x in cl) / len(cl)
            if abs(y_mid - cy) <= y_thr:
                cl.append(w)
                placed = True
                break
        if not placed:
            line_clusters.append([w])

    lines = []
    for cl in line_clusters:
        cl = sorted(cl, key=lambda x: x["bbox"][0])
        text = _normalize_text(" ".join(x["text"] for x in cl))
        if not _is_text_like(text):
            continue
        x1 = min(x["bbox"][0] for x in cl)
        y1 = min(x["bbox"][1] for x in cl)
        x2 = max(x["bbox"][0] + x["bbox"][2] for x in cl)
        y2 = max(x["bbox"][1] + x["bbox"][3] for x in cl)
        conf = sum(x["confidence"] for x in cl) / len(cl)
        lines.append({"text": text, "bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": round(conf, 3)})

    lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return lines


def ocr_lines(image_bgr) -> list[dict[str, Any]]:
    reader = get_reader()
    raw_candidates: list[dict[str, Any]] = []

    for _, variant, scale in build_ocr_variants(image_bgr):
        result = reader.readtext(variant, detail=1, paragraph=False)
        for box, text, conf in result:
            text = _normalize_text(str(text))
            conf = float(conf)
            if conf < OCR_CONF_THRESHOLD or not _is_text_like(text):
                continue
            x, y, w, h = _bbox_from_quad(box)
            if scale != 1.0:
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            if w * h < OCR_MIN_BOX_AREA:
                continue
            raw_candidates.append({"text": text, "bbox": [x, y, w, h], "confidence": conf})

    merged_words = _merge_candidates(raw_candidates)
    return _merge_words_into_lines(merged_words)


def draw_lines(image_bgr, lines: list[dict[str, Any]]):
    out = image_bgr.copy()
    for line in lines:
        x, y, w, h = line["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


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


def _extract_after_label(lines: list[str], labels: list[str]) -> str | None:
    for i, line in enumerate(lines):
        for lbl in labels:
            if lbl in line:
                rest = _normalize_text(line.split(lbl, 1)[-1]).strip(" :")
                if len(rest) >= 2:
                    return rest
                if i + 1 < len(lines):
                    nxt = _normalize_text(lines[i + 1]).strip(" :")
                    if len(nxt) >= 2:
                        return nxt
    return None


def heuristic_extract(ocr_lines_text: list[str]) -> dict[str, Any]:
    lines = [x for x in (_line_norm(t) for t in ocr_lines_text) if x]
    joined = " ".join(lines)

    surname = _extract_after_label(lines, ["ФАМИЛ", "SURNAME"])
    name = _extract_after_label(lines, ["ИМЯ", "NAME"])
    patronymic = _extract_after_label(lines, ["ОТЧЕСТ", "PATRONYMIC", "MIDDLE"])

    full_name = None
    if surname or name or patronymic:
        parts = [p for p in [surname, name, patronymic] if p]
        if parts:
            full_name = _normalize_text(" ".join(parts))

    if not full_name:
        stop_words = {
            "ВОДИТЕЛЬСКОЕ", "УДОСТОВЕРЕНИЕ", "ПАСПОРТ", "РЕСПУБЛИКА", "ФЕДЕРАЦИЯ", "РОССИЙСКАЯ",
            "ДАТА", "ВЫДАЧИ", "КОД", "ПОДРАЗДЕЛЕНИЯ", "МВД", "ПО", "ГОРОД", "ГОР",
        }
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
        "ocr_engine": "easyocr_multipass",
        "use_gpu": USE_GPU,
        "ocr_conf_threshold": OCR_CONF_THRESHOLD,
        "ocr_min_box_area": OCR_MIN_BOX_AREA,
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
            "engine": "easyocr_multipass",
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

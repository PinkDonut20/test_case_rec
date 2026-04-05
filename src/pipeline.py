from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import HTTPException, UploadFile

from src.config import Settings
from src.extraction import ExtractorService
from src.ocr import OCRService


def draw_lines(image_bgr, lines: list[dict[str, Any]]):
    out = image_bgr.copy()
    for line in lines:
        bbox = line.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


class DocumentPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ocr = OCRService(settings)
        self.extractor = ExtractorService(settings)

    async def process_upload(self, file: UploadFile) -> dict[str, Any]:
        suffix = Path(file.filename or "input.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            raw = await file.read()
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        image = cv2.imread(str(tmp_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        lines, ocr_error = self.ocr.run(image)
        line_texts = [x.get("text", "") for x in lines if x.get("text")]
        full_text = "\n".join(line_texts)

        fields, extractor_mode, extractor_error = self.extractor.extract(line_texts)

        stem = Path(file.filename or "input").stem
        annotated_path = self.settings.output_dir / f"{stem}_annotated.jpg"
        json_path = self.settings.output_dir / f"{stem}_result.json"

        cv2.imwrite(str(annotated_path), draw_lines(image, lines))

        payload = {
            "input_image": file.filename,
            "annotated_image": str(annotated_path),
            "ocr": {
                "engine": self.settings.ocr_backend,
                "lines": lines,
                "full_text": full_text,
                "lines_count": len(lines),
                "error": ocr_error,
            },
            "fields": fields,
            "extractor": {
                "mode": extractor_mode,
                "error": extractor_error,
                "config_mode": self.settings.extractor_mode,
                "hf_model": self.settings.hf_model if extractor_mode == "hf" else None,
                "llm_model": self.settings.llm_model if extractor_mode == "api" else None,
                "llm_base_url": self.settings.llm_base_url if extractor_mode == "api" else None,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import easyocr
from transformers import pipeline

from src.config import Settings


@dataclass
class OCRLine:
    text: str
    bbox: list[int]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "bbox": self.bbox, "confidence": self.confidence}


class EasyOCRBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.reader = easyocr.Reader(["ru", "en"], gpu=settings.use_gpu)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).strip().split())

    @staticmethod
    def _bbox_from_quad(box) -> list[int]:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x, y = min(xs), min(ys)
        return [x, y, max(xs) - x, max(ys) - y]

    @staticmethod
    def _build_variants(image_bgr):
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        sharp = cv2.filter2D(image_bgr, -1, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        up = cv2.resize(image_bgr, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        return [(image_bgr, 1.0), (gray, 1.0), (clahe, 1.0), (sharp, 1.0), (up, 1.5)]

    def run(self, image_bgr) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for variant, scale in self._build_variants(image_bgr):
            result = self.reader.readtext(variant, detail=1, paragraph=True)
            for box, text, conf in result:
                text = self._normalize(text)
                conf = float(conf)
                if not text or conf < self.settings.ocr_conf_threshold:
                    continue
                x, y, w, h = self._bbox_from_quad(box)
                if scale != 1.0:
                    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                if w * h < self.settings.ocr_min_box_area:
                    continue
                lines.append(OCRLine(text=text, bbox=[x, y, w, h], confidence=round(conf, 3)))

        lines.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
        dedup: list[OCRLine] = []
        seen = set()
        for line in lines:
            key = (line.text.upper(), line.bbox[0] // 20, line.bbox[1] // 20)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(line)
        return dedup


class LightOnOCRBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = pipeline(
            task="image-text-to-text",
            model=settings.lighton_model,
            device=0 if settings.use_gpu else -1,
        )

    def run(self, image_bgr) -> list[OCRLine]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.pipe(image_rgb, max_new_tokens=768)
        text = ""
        if isinstance(result, list) and result:
            row = result[0]
            text = row.get("generated_text", "") if isinstance(row, dict) else str(row)
        else:
            text = str(result)

        clean_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not clean_lines:
            return []

        h, w = image_bgr.shape[:2]
        step = max(1, h // max(1, len(clean_lines) + 2))
        lines: list[OCRLine] = []
        for i, ln in enumerate(clean_lines):
            y = min(h - 20, (i + 1) * step)
            lines.append(OCRLine(text=ln, bbox=[10, y, max(20, w - 20), min(40, step)], confidence=0.95))
        return lines


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._primary = None
        self._fallback = None

    def _get_primary(self):
        if self._primary is not None:
            return self._primary
        if self.settings.ocr_backend == "lighton":
            self._primary = LightOnOCRBackend(self.settings)
        else:
            self._primary = EasyOCRBackend(self.settings)
        return self._primary

    def _get_fallback(self):
        if self._fallback is None:
            self._fallback = EasyOCRBackend(self.settings)
        return self._fallback

    def run(self, image_bgr) -> tuple[list[dict[str, Any]], str | None]:
        err = None
        try:
            lines = [x.as_dict() for x in self._get_primary().run(image_bgr)]
            if lines:
                return lines, None
        except Exception as e:
            err = f"primary_ocr_error: {e}"

        fallback = [x.as_dict() for x in self._get_fallback().run(image_bgr)]
        return fallback, err

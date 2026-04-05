from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from openai import OpenAI
from transformers import pipeline

from src.config import Settings


class ExtractorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._hf_pipe = None

    def _get_hf_pipe(self):
        if self._hf_pipe is None:
            self._hf_pipe = pipeline("text-generation", model=self.settings.hf_model)
        return self._hf_pipe

    @staticmethod
    def _line_norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().upper())

    @staticmethod
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

    @staticmethod
    def _extract_after_label(lines: list[str], labels: list[str]) -> str | None:
        for i, line in enumerate(lines):
            for lbl in labels:
                if lbl in line:
                    rest = " ".join(line.split(lbl, 1)[-1].strip(" :").split())
                    if len(rest) >= 2:
                        return rest
                    if i + 1 < len(lines):
                        nxt = " ".join(lines[i + 1].strip(" :").split())
                        if len(nxt) >= 2:
                            return nxt
        return None

    def heuristic_extract(self, line_texts: list[str]) -> dict[str, Any]:
        lines = [x for x in (self._line_norm(t) for t in line_texts) if x]
        joined = " ".join(lines)

        surname = self._extract_after_label(lines, ["ФАМИЛ", "SURNAME"])
        name = self._extract_after_label(lines, ["ИМЯ", "NAME"])
        patronymic = self._extract_after_label(lines, ["ОТЧЕСТ", "PATRONYMIC", "MIDDLE"])

        full_name = None
        if surname or name or patronymic:
            full_name = " ".join([p for p in [surname, name, patronymic] if p])

        if not full_name:
            stop_words = {
                "ВОДИТЕЛЬСКОЕ",
                "УДОСТОВЕРЕНИЕ",
                "ПАСПОРТ",
                "РЕСПУБЛИКА",
                "ФЕДЕРАЦИЯ",
                "РОССИЙСКАЯ",
                "ДАТА",
                "ВЫДАЧИ",
                "КОД",
                "ПОДРАЗДЕЛЕНИЯ",
                "МВД",
                "ПО",
                "ГОРОД",
                "ГОР",
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
        for line in lines:
            birth_date = self._parse_date(line)
            if birth_date:
                break

        document_number = None
        for p in [r"\b(\d{2}\s?\d{2}\s?\d{6})\b", r"\b(\d{9,12})\b"]:
            m = re.search(p, joined)
            if m:
                document_number = re.sub(r"\s+", " ", m.group(1)).strip()
                break

        return {"full_name": full_name, "birth_date": birth_date, "document_number": document_number}

    @staticmethod
    def _normalize_doc_number(value: str | None) -> str | None:
        if not value:
            return None
        digits = re.sub(r"\D", "", value)
        if len(digits) == 10:
            return f"{digits[:2]} {digits[2:4]} {digits[4:]}"
        if 9 <= len(digits) <= 12:
            return digits
        return value.strip()

    @staticmethod
    def _invalid_full_name(value: str | None) -> bool:
        if not value:
            return True
        v = re.sub(r"\s+", " ", value.strip().upper())
        banned = {"ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ", "ПАСПОРТ", "РОССИЙСКАЯ ФЕДЕРАЦИЯ", "МВД"}
        if v in banned:
            return True
        return any(t in v.split() for t in ["ВОДИТЕЛЬСКОЕ", "УДОСТОВЕРЕНИЕ", "ПАСПОРТ", "ФЕДЕРАЦИЯ", "МВД"])

    def _postprocess_fields(self, fields: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        out = dict(fields)
        out["document_number"] = self._normalize_doc_number(out.get("document_number"))
        if self._invalid_full_name(out.get("full_name")):
            out["full_name"] = fallback.get("full_name")
        if not out.get("birth_date"):
            out["birth_date"] = fallback.get("birth_date")
        if not out.get("document_number"):
            out["document_number"] = fallback.get("document_number")
        return out

    @staticmethod
    def _extract_first_json(text: str) -> dict[str, Any] | None:
        m = re.search(r"\{[\s\S]*?\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            raw = m.group(0)
            depth, end_idx = 0, 0
            for i, ch in enumerate(raw):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            try:
                return json.loads(raw[:end_idx])
            except json.JSONDecodeError:
                return None

    def hf_extract(self, line_texts: list[str]) -> tuple[dict[str, Any] | None, str | None]:
        pipe = self._get_hf_pipe()
        full_text = "\n".join(line_texts)
        prompt = (
            "Извлеки поля full_name, birth_date, document_number из OCR. "
            "Не используй заголовки документа как ФИО. Верни только JSON.\n"
            f"OCR lines: {json.dumps(line_texts, ensure_ascii=False)}\n"
            f"OCR full_text: {full_text}"
        )
        out = pipe(prompt, max_new_tokens=160, do_sample=False, return_full_text=False)
        txt = out[0].get("generated_text", "") if isinstance(out, list) else str(out)
        parsed = self._extract_first_json(txt)
        if not parsed:
            return None, f"HF model returned non-JSON: {txt[:200]}"
        return {
            "full_name": parsed.get("full_name"),
            "birth_date": parsed.get("birth_date"),
            "document_number": parsed.get("document_number"),
        }, None

    def api_extract(self, line_texts: list[str]) -> dict[str, Any]:
        if not self.settings.llm_base_url:
            raise ValueError("LLM_BASE_URL is empty")

        client = OpenAI(base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key or "dummy")
        prompt = (
            "Извлеки из OCR текста поля full_name, birth_date, document_number и верни только JSON. "
            "Нельзя использовать заголовки документа как ФИО. Если не уверен — верни null.\n"
            f"OCR lines: {json.dumps(line_texts, ensure_ascii=False)}\n"
            f"OCR full_text: {' '.join(line_texts)}"
        )
        resp = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": "Ответ только JSON без markdown и пояснений."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        parsed = self._extract_first_json(content)
        if not parsed:
            raise ValueError(f"API model did not return JSON: {content[:300]}")
        return {
            "full_name": parsed.get("full_name"),
            "birth_date": parsed.get("birth_date"),
            "document_number": parsed.get("document_number"),
        }

    def extract(self, line_texts: list[str]) -> tuple[dict[str, Any], str, str | None]:
        fallback = self.heuristic_extract(line_texts)

        try:
            if self.settings.extractor_mode == "api":
                raw = self.api_extract(line_texts)
                return self._postprocess_fields(raw, fallback), "api", None
            if self.settings.extractor_mode == "hf":
                raw, err = self.hf_extract(line_texts)
                if raw is None:
                    raise ValueError(err or "hf extraction failed")
                return self._postprocess_fields(raw, fallback), "hf", None
            if self.settings.extractor_mode == "hybrid":
                try:
                    raw = self.api_extract(line_texts)
                    return self._postprocess_fields(raw, fallback), "api", None
                except Exception:
                    raw, err = self.hf_extract(line_texts)
                    if raw is not None:
                        return self._postprocess_fields(raw, fallback), "hf", None
                    raise ValueError(err or "hybrid extraction failed")
            return fallback, "heuristic", None
        except Exception as e:
            return fallback, "heuristic_fallback", str(e)

import io
import json
import os
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, filedialog, messagebox, ttk
import tkinter as tk

import requests
from PIL import Image, ImageDraw, ImageTk

API_URL = os.getenv("OCR_API_URL", "http://localhost:8000/process")
UI_OUTPUT_DIR = Path(os.getenv("UI_OUTPUT_DIR", "ui_outputs"))
UI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class OCRDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Doc OCR Desktop")
        self.root.geometry("1200x800")

        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None
        self.current_payload: dict | None = None
        self.current_json_path: Path | None = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=BOTH, expand=True)

        top_row = ttk.Frame(main)
        top_row.pack(fill=X)

        ttk.Button(top_row, text="Загрузить изображение", command=self.load_image).pack(side=LEFT, padx=4)
        ttk.Button(top_row, text="Распознать", command=self.process_image).pack(side=LEFT, padx=4)
        ttk.Button(top_row, text="Сохранить JSON", command=self.save_json).pack(side=LEFT, padx=4)
        ttk.Button(top_row, text="Очистить", command=self.clear).pack(side=LEFT, padx=4)

        self.status_var = tk.StringVar(value=f"Готово. API: {API_URL}")
        ttk.Label(main, textvariable=self.status_var).pack(fill=X, pady=(8, 10))

        body = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        body.pack(fill=BOTH, expand=True)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=1)

        ttk.Label(left, text="Исходное изображение").pack(anchor="w")
        self.input_canvas = tk.Canvas(left, bg="#111111", height=350)
        self.input_canvas.pack(fill=BOTH, expand=True, pady=(4, 10))

        ttk.Label(left, text="Размеченное изображение (OCR боксы)").pack(anchor="w")
        self.output_canvas = tk.Canvas(left, bg="#111111", height=350)
        self.output_canvas.pack(fill=BOTH, expand=True, pady=(4, 0))

        notebook = ttk.Notebook(right)
        notebook.pack(fill=BOTH, expand=True)

        tab_fields = ttk.Frame(notebook)
        tab_ocr = ttk.Frame(notebook)
        tab_json = ttk.Frame(notebook)
        notebook.add(tab_fields, text="Поля")
        notebook.add(tab_ocr, text="OCR текст")
        notebook.add(tab_json, text="Полный JSON")

        self.fields_text = tk.Text(tab_fields, wrap="word")
        self.fields_text.pack(fill=BOTH, expand=True)

        self.ocr_text = tk.Text(tab_ocr, wrap="word")
        self.ocr_text.pack(fill=BOTH, expand=True)

        self.json_text = tk.Text(tab_json, wrap="word")
        self.json_text.pack(fill=BOTH, expand=True)

    @staticmethod
    def _draw_boxes(image: Image.Image, lines: list[dict]) -> Image.Image:
        out = image.convert("RGB").copy()
        draw = ImageDraw.Draw(out)
        for line in lines:
            bbox = line.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 140), width=3)
        return out

    @staticmethod
    def _resize_for_canvas(image: Image.Image, canvas: tk.Canvas) -> ImageTk.PhotoImage:
        cw = max(200, canvas.winfo_width())
        ch = max(200, canvas.winfo_height())
        img = image.copy()
        img.thumbnail((cw - 10, ch - 10))
        return ImageTk.PhotoImage(img)

    def _render_image(self, canvas: tk.Canvas, image: Image.Image, tag: str):
        canvas.update_idletasks()
        photo = self._resize_for_canvas(image, canvas)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo, anchor="center", tags=tag)
        setattr(self, f"_{tag}_photo", photo)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_image = Image.open(path).convert("RGB")
        self._render_image(self.input_canvas, self.input_image, "input")
        self.status_var.set(f"Загружено: {Path(path).name}")

    def process_image(self):
        if self.input_image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение")
            return

        self.status_var.set("Отправка в API...")
        self.root.update_idletasks()

        buf = io.BytesIO()
        self.input_image.save(buf, format="JPEG")
        buf.seek(0)

        try:
            response = requests.post(
                API_URL,
                files={"file": ("document.jpg", buf, "image/jpeg")},
                timeout=240,
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось подключиться к API:\n{e}")
            self.status_var.set("Ошибка подключения к API")
            return

        if response.status_code != 200:
            messagebox.showerror("Ошибка API", f"Код: {response.status_code}\n{response.text[:500]}")
            self.status_var.set("Ошибка API")
            return

        payload = response.json()
        self.current_payload = payload

        lines = payload.get("ocr", {}).get("lines", [])
        self.output_image = self._draw_boxes(self.input_image, lines)
        self._render_image(self.output_canvas, self.output_image, "output")

        fields = payload.get("fields", {})
        full_text = payload.get("ocr", {}).get("full_text", "")

        self.fields_text.delete("1.0", END)
        self.fields_text.insert("1.0", json.dumps(fields, ensure_ascii=False, indent=2))

        self.ocr_text.delete("1.0", END)
        self.ocr_text.insert("1.0", full_text)

        self.json_text.delete("1.0", END)
        self.json_text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_json_path = UI_OUTPUT_DIR / f"result_{stamp}.json"
        self.current_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines_count = payload.get("ocr", {}).get("lines_count", len(lines))
        self.status_var.set(f"Готово: строк OCR={lines_count}, JSON={self.current_json_path.name}")

    def save_json(self):
        if not self.current_payload:
            messagebox.showinfo("Инфо", "Сначала выполните распознавание")
            return

        target = filedialog.asksaveasfilename(
            title="Сохранить JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=self.current_json_path.name if self.current_json_path else "result.json",
        )
        if not target:
            return

        Path(target).write_text(json.dumps(self.current_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"JSON сохранён: {Path(target).name}")

    def clear(self):
        self.input_image = None
        self.output_image = None
        self.current_payload = None
        self.current_json_path = None

        self.input_canvas.delete("all")
        self.output_canvas.delete("all")
        self.fields_text.delete("1.0", END)
        self.ocr_text.delete("1.0", END)
        self.json_text.delete("1.0", END)
        self.status_var.set(f"Очищено. API: {API_URL}")


def main():
    root = tk.Tk()
    app = OCRDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

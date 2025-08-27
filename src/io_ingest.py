import os, csv, pathlib
from typing import List, Optional
from docx import Document as DocxDocument
import pdfplumber

def _read_txt(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_docx(p: str) -> str:
    doc = DocxDocument(p)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def _read_pdf(p: str) -> str:
    out = []
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)

def _read_csv(p: str, text_column: str = "text") -> List[str]:
    rows = []
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if text_column not in reader.fieldnames:
            raise ValueError(f"CSV must have a '{text_column}' column.")
        for r in reader:
            rows.append(r.get(text_column, "") or "")
    return rows

def load_texts(path: str, csv_text_column: str = "text") -> List[str]:
    """
    If 'path' is a file: returns [text] or list of texts (CSV).
    If 'path' is a directory: returns texts from all supported files inside.
    Supported: .txt, .docx, .pdf, .csv
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    def read_one(fp: pathlib.Path) -> List[str]:
        suf = fp.suffix.lower()
        if suf == ".txt":
            return [_read_txt(str(fp))]
        if suf == ".docx":
            return [_read_docx(str(fp))]
        if suf == ".pdf":
            return [_read_pdf(str(fp))]
        if suf == ".csv":
            return _read_csv(str(fp), csv_text_column)
        # ignore unknowns
        return []

    if p.is_file():
        return read_one(p)
    else:
        texts = []
        for fp in sorted(p.iterdir()):
            texts.extend(read_one(fp))
        return [t for t in texts if t and t.strip()]

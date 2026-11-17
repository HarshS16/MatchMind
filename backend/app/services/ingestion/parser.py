"""Resume text extraction for PDF and DOCX files."""

import io
import logging

logger = logging.getLogger(__name__)


class UnsupportedFileError(ValueError):
    pass


def extract_pdf_text(data: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def extract_docx_text(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells if cell.text.strip()))
    return "\n".join(parts)


def extract_resume_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return extract_pdf_text(data)
    if ext == "docx":
        return extract_docx_text(data)
    raise UnsupportedFileError(f"Unsupported file type: .{ext}")

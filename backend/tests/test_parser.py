import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingestion.parser import UnsupportedFileError, extract_resume_text  # noqa: E402


def _minimal_docx(paragraphs: list[str]) -> bytes:
    """Build a minimal .docx (zip) containing document.xml with given paragraphs."""
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document {ns}><w:body>{body}</w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                   "</Relationships>")
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def test_docx_extraction():
    data = _minimal_docx(["Jane Doe", "Software Engineer", "Skills: Python, AWS"])
    text = extract_resume_text("resume.docx", data)
    assert "Jane Doe" in text
    assert "Python" in text


def test_unsupported_extension():
    try:
        extract_resume_text("resume.txt", b"hello")
        assert False, "expected UnsupportedFileError"
    except UnsupportedFileError:
        pass

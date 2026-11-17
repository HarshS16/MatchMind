"""Text cleaning and lightweight structured extraction for resumes and job descriptions."""

import re
from dataclasses import dataclass, field

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
URL_RE = re.compile(r"https?://\S+")

# Common resume section headers used for chunking/section identification.
SECTION_HEADERS = [
    "summary", "objective", "profile", "skills", "technical skills",
    "experience", "work experience", "employment", "projects", "project",
    "education", "certifications", "awards", "publications",
]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def pseudonymize(text: str) -> str:
    """Remove obvious PII (emails, phones, URLs) from stored resume text."""
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = URL_RE.sub("[url]", text)
    return text


def extract_name(text: str) -> str | None:
    for line in text.split("\n")[:15]:
        line = line.strip()
        if line and not EMAIL_RE.search(line) and not PHONE_RE.search(line) and len(line.split()) <= 5:
            return line
    return None


def extract_skills(text: str) -> list[str]:
    """Simple skill extraction: match known tech terms against a curated vocabulary."""
    vocab = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "sql", "nosql", "pytorch", "tensorflow", "keras", "scikit-learn",
        "nlp", "computer vision", "deep learning", "machine learning",
        "react", "angular", "vue", "node.js", "express", "django", "flask",
        "fastapi", "spring", "docker", "kubernetes", "aws", "gcp", "azure",
        "git", "linux", "postgresql", "mysql", "mongodb", "redis",
        "spark", "hadoop", "airflow", "kafka", "pandas", "numpy",
        "transformers", "hugging face", "llm", "rag", "langchain",
    ]
    def display(term: str) -> str:
        return " ".join(w[:1].upper() + w[1:] for w in term.split())

    lower = text.lower()
    found = []
    for term in vocab:
        if re.search(rf"\b{re.escape(term)}\b", lower):
            found.append(display(term))
    return sorted(set(found), key=str.lower)


def extract_experience_years(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


SECTION_SPLIT_RE = re.compile(
    r"(?im)^\s*(summary|objective|profile|skills|technical skills|experience|"
    r"work experience|employment( history)?|projects?|education|certifications?|"
    r"awards?|publications?)\s*:?\s*$"
)


@dataclass
class DocumentChunk:
    section: str
    text: str
    start: int = 0
    end: int = 0


def split_sections(text: str, max_chars: int = 4000) -> list[DocumentChunk]:
    """Split resume text into section-level chunks (fallback: whole doc)."""
    matches = list(SECTION_SPLIT_RE.finditer(text))
    chunks: list[DocumentChunk] = []
    if not matches:
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars]
            if chunk.strip():
                chunks.append(DocumentChunk(section="document", text=chunk.strip(), start=i, end=i + len(chunk)))
        return chunks or [DocumentChunk(section="document", text=text.strip())]

    # Preamble before the first header
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(DocumentChunk(section="header", text=preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body:
            continue
        section = m.group(1).lower().replace(" ", "_")
        # Further split oversized sections
        for j in range(0, len(body), max_chars):
            part = body[j:j + max_chars]
            if part.strip():
                chunks.append(DocumentChunk(section=section, text=part.strip(), start=m.end() + j, end=m.end() + j + len(part)))
    return chunks


@dataclass
class JobRequirements:
    title: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_min: int | None = None
    experience_max: int | None = None
    responsibilities: str = ""
    raw: str = ""

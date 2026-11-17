import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.preprocessing.text import (  # noqa: E402
    clean_text,
    extract_experience_years,
    extract_skills,
    pseudonymize,
    split_sections,
)


def test_clean_text():
    assert clean_text("hello\t world\n\n\n\nfoo") == "hello world\n\nfoo"


def test_pseudonymize():
    text = "Contact me at john@example.com or +1 555 123 4567, site https://x.com"
    out = pseudonymize(text)
    assert "john@example.com" not in out
    assert "555" not in out
    assert "https://x.com" not in out


def test_extract_skills():
    text = "Experienced with Python, PyTorch and Docker. Also knows React and SQL."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Docker" in skills


def test_extract_experience_years():
    assert extract_experience_years("5+ years of experience") == 5.0
    assert extract_experience_years("no duration here") is None


def test_split_sections():
    text = (
        "John Doe\n"
        "Summary\nML engineer\n"
        "Experience\n2020-2024 Acme Corp, ML role\n"
        "Education\nBS Computer Science\n"
    )
    chunks = split_sections(text)
    sections = {c.section for c in chunks}
    assert "summary" in sections
    assert "experience" in sections
    assert "education" in sections

"""Context builder for candidate-centric semantic indexing."""
from typing import Any, Dict


def _to_text(value: Any, default: str = "N/A") -> str:
    """Convert nullable/unknown values to safe display text."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _summary_text(resume_text: Any, max_chars: int = 1000) -> str:
    """Return first max_chars from resume text for context summary."""
    text = _to_text(resume_text, default="")
    if not text:
        return "N/A"
    return text[:max_chars]


def build_candidate_context(row: Dict[str, Any]) -> str:
    """
    Build structured context text for context-aware candidate indexing.

    IMPORTANT: This intentionally stores meaning-rich structured context,
    not raw resume text.
    """
    candidate_name = _to_text(row.get("candidatename"))
    designation = _to_text(row.get("designation"))
    mastercategory = _to_text(row.get("mastercategory"))
    domain = _to_text(row.get("domain"))
    experience = _to_text(row.get("experience"))
    skills = _to_text(row.get("skillset"))
    education = _to_text(row.get("education"))
    summary = _summary_text(row.get("resume_text"), max_chars=1000)

    return (
        "Entity: Candidate\n\n"
        f"Name: {candidate_name}\n"
        f"Role: {designation}\n"
        f"Category: {mastercategory}\n"
        f"Domain: {domain}\n\n"
        f"Experience: {experience} years\n\n"
        "Skills:\n"
        f"{skills}\n\n"
        "Education:\n"
        f"{education}\n\n"
        "Summary:\n"
        f"{summary}"
    )


"""Context builder for candidate-centric semantic indexing."""
import re
from typing import Any, Dict, List, Optional, Sequence


def _non_empty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coerce_experience_years(experience: Any) -> Optional[int]:
    """Return integer years if parseable; else None."""
    if experience is None:
        return None
    if isinstance(experience, bool):
        return None
    if isinstance(experience, (int, float)):
        return int(experience)
    s = str(experience).strip()
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*years?\s*$", s, re.IGNORECASE)
    if m:
        return int(float(m.group(1)))
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return int(float(s))
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        return int(float(m2.group(1)))
    return None


def _join_list(items: Any) -> Optional[str]:
    if not items:
        return None
    if isinstance(items, str):
        return _non_empty_str(items)
    if not isinstance(items, Sequence):
        return None
    parts: List[str] = []
    for x in items:
        t = _non_empty_str(x)
        if t:
            parts.append(t)
    return ", ".join(parts) if parts else None


def _projects_summary(projects: Any, max_items: int = 5) -> Optional[str]:
    if not projects:
        return None
    if isinstance(projects, str):
        return _non_empty_str(projects)
    if not isinstance(projects, Sequence):
        return None
    items: List[str] = []
    for p in list(projects)[:max_items]:
        if isinstance(p, dict):
            name = _non_empty_str(p.get("name") or p.get("title"))
            desc = _non_empty_str(p.get("description") or p.get("summary"))
            chunk = name or desc
            if name and desc and desc != name:
                chunk = f"{name}: {desc}"
            if chunk:
                items.append(chunk)
        else:
            t = _non_empty_str(p)
            if t:
                items.append(t)
    return ", ".join(items) if items else None


def build_context(profile: Dict[str, Any]) -> str:
    """
    Convert structured candidate data into a natural-language context string
    for embedding and semantic search.

    Supported keys: role, experience, skills, tools, domain, education,
    certifications, projects, location.

    Only non-empty fields are included. Output is a single clean paragraph.
    """
    if not profile:
        return ""

    sentences: List[str] = []

    role = _non_empty_str(profile.get("role"))
    if role:
        sentences.append(f"Candidate is a {role}.")

    exp = profile.get("experience")
    years = _coerce_experience_years(exp)
    if years is not None:
        sentences.append(f"Has {years} years of experience.")
    else:
        exp_s = _non_empty_str(exp)
        if exp_s:
            sentences.append(f"Professional experience includes {exp_s}.")

    skills = _join_list(profile.get("skills"))
    if skills:
        sentences.append(f"Skilled in {skills}.")

    tools = _join_list(profile.get("tools"))
    if tools:
        sentences.append(f"Experienced with tools like {tools}.")

    domain = _non_empty_str(profile.get("domain"))
    if domain:
        sentences.append(f"Worked in {domain} domain.")

    education = _non_empty_str(profile.get("education"))
    if education:
        # Keep bounded for embedding payload size
        if len(education) > 2000:
            education = education[:2000].rstrip() + "..."
        sentences.append(f"Education includes {education}.")

    certs = _join_list(profile.get("certifications"))
    if certs:
        sentences.append(f"Certified in {certs}.")

    proj = _projects_summary(profile.get("projects"))
    if proj:
        sentences.append(f"Worked on projects involving {proj}.")

    loc = _non_empty_str(profile.get("location"))
    if loc:
        sentences.append(f"Based in {loc}.")

    return " ".join(sentences).strip()


def row_to_index_context_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a resume_metadata-style DB row dict to the profile expected by build_context.

    Excludes fields that must not appear in embedding text (id, resume_text, PII, etc.).
    """
    profile: Dict[str, Any] = {}

    role = _non_empty_str(row.get("designation")) or _non_empty_str(row.get("jobrole"))
    if role:
        profile["role"] = role

    exp = row.get("experience")
    if exp is not None and _non_empty_str(exp) is not None:
        profile["experience"] = exp

    skillset = row.get("skillset")
    skills_list: List[str] = []
    if skillset is not None:
        if isinstance(skillset, list):
            skills_list = [str(s).strip() for s in skillset if str(s).strip()]
        else:
            skills_list = [
                s.strip() for s in str(skillset).split(",") if s.strip()
            ]
    if skills_list:
        profile["skills"] = skills_list

    domain = _non_empty_str(row.get("category")) or _non_empty_str(
        row.get("mastercategory")
    )
    if domain:
        profile["domain"] = domain

    loc = _non_empty_str(row.get("location"))
    if loc:
        profile["location"] = loc

    edu = _non_empty_str(row.get("education"))
    if edu:
        profile["education"] = edu

    return profile


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


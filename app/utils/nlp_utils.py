"""
NLP utilities for AI search: stemming, stop-word removal, tokenization, and name-query detection.
Used to improve skill/designation matching and to skip LLM for obvious name-only queries.
"""
import re
from typing import List, Set, Optional, Dict, Tuple


# Whole-word replacements before LLM parse (complements AI_SEARCH_PROMPT spelling rules).
_QUERY_TYPO_FIXES: Tuple[Tuple[str, str], ...] = (
    (r"\bdevelopor\b", "developer"),
    (r"\bdevaloper\b", "developer"),
    (r"\bpythan\b", "python"),
    (r"\bpytaan\b", "python"),
    (r"\bpythoon\b", "python"),
)


def fix_common_query_typos(text: str) -> str:
    """
    Fix frequent keyboard/OCR typos in recruiter search queries (whole words only).
    Examples: python developor -> python developer; pythan devaloper -> python developer.
    """
    if not text or not text.strip():
        return text
    out = text
    for pattern, repl in _QUERY_TYPO_FIXES:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Stop words for search/skill matching (exclude from token overlap when useful)
# ---------------------------------------------------------------------------
STOP_WORDS_SEARCH: Set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "years", "year", "experience", "exp", "plus", "minimum", "maximum",
    "senior", "lead", "jr", "sr", "junior", "principal", "staff", "level",
    "engineer", "developer", "manager", "analyst", "specialist", "consultant",
    # Keep these minimal so we don't over-filter role phrases; expand as needed
}


def _ensure_nltk() -> bool:
    """Ensure NLTK stopwords and PorterStemmer are available. Returns True if usable."""
    try:
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        return True
    except Exception:
        return False


def get_stop_words() -> Set[str]:
    """Return stop words for search (built-in + optional NLTK English)."""
    out = set(STOP_WORDS_SEARCH)
    if _ensure_nltk():
        try:
            from nltk.corpus import stopwords
            out.update(w.lower() for w in stopwords.words("english"))
        except Exception:
            pass
    return out


_stemmer = None


def _get_stemmer():
    """Lazy-load NLTK PorterStemmer. Returns None if NLTK not available."""
    global _stemmer
    if _stemmer is not None:
        return _stemmer
    if not _ensure_nltk():
        return None
    try:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
        return _stemmer
    except Exception:
        return None


def stem_word(word: str) -> str:
    """Return stemmed form of a single word. Falls back to lowercased word if no stemmer."""
    if not word or not word.strip():
        return ""
    w = word.strip().lower()
    ps = _get_stemmer()
    if ps is None:
        return w
    try:
        return ps.stem(w)
    except Exception:
        return w


def stem_tokens(tokens: List[str]) -> List[str]:
    """Stem a list of tokens. Returns list of stemmed strings (no empties)."""
    result = []
    for t in tokens:
        s = stem_word(t)
        if s:
            result.append(s)
    return result


def tokenize_for_matching(text: Optional[str]) -> List[str]:
    """
    Tokenize text for matching: split on whitespace and punctuation but keep
    tech tokens like 'c#', '.net' as single units where possible.
    Returns list of lowercase tokens (no empties).
    """
    if not text or not str(text).strip():
        return []
    text = str(text).strip().lower()
    # Split on whitespace first
    parts = re.split(r"\s+", text)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Keep known compound tech tokens (e.g. c#, .net) as one token
        if part in (".net", "c#", "c++", "f#", "r.", "vb.net"):
            tokens.append(part.replace(".", "").replace("#", "sharp").replace("+", "plus"))
            continue
        if part.endswith("#") or part.startswith("."):
            # e.g. "c#" or ".net"
            clean = re.sub(r"[^\w]", "", part) or part
            if clean:
                tokens.append(clean)
            continue
        # Split on non-alphanumeric (except hyphen within word), keep words
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", part):
            if token:
                tokens.append(token)
    return tokens


def normalize_tokens_for_match(text: Optional[str], remove_stop_words: bool = True) -> List[str]:
    """
    Normalize text for matching: tokenize, optionally remove stop words, stem.
    Returns list of stemmed tokens suitable for overlap comparison.
    """
    tokens = tokenize_for_matching(text)
    if not tokens:
        return []
    if remove_stop_words:
        stop = get_stop_words()
        tokens = [t for t in tokens if t not in stop]
    return stem_tokens(tokens)


def stemmed_skill_overlap_ratio(required_skills: List[str], candidate_skills: List[str]) -> float:
    """
    Compute overlap ratio between required and candidate skills using stemmed tokens.
    Returns value in [0.0, 1.0]: proportion of required skills that have at least
    one stemmed match in candidate skills.
    """
    if not required_skills:
        return 1.0
    req_stems: Set[str] = set()
    for s in required_skills:
        req_stems.update(normalize_tokens_for_match(s, remove_stop_words=False))
    cand_stems: Set[str] = set()
    for s in candidate_skills:
        cand_stems.update(normalize_tokens_for_match(s, remove_stop_words=False))
    if not req_stems:
        # Required skills produced no tokens — do not treat as 100% match
        return 0.0 if required_skills else 1.0
    matched = len(req_stems & cand_stems)
    return matched / len(req_stems)


# Keywords that strongly suggest a role/skill query (not a person name)
_ROLE_SKILL_KEYWORDS = frozenset({
    "developer", "engineer", "manager", "analyst", "qa", "python", "java", "sql",
    "selenium", "automation", "devops", "aws", "azure", "experience", "years",
    "and", "or", "with", "skills", "testing", "frontend", "backend", "fullstack",
    "data", "software", "senior", "lead", "junior", "principal", "staff",
    "project", "product", "scrum", "agile", "designer", "architect", "admin",
    # Treat common tools/techs as non-name so queries like "power bi" don't hit name search
    "power", "bi", "tableau", "excel", "ssis", "ssrs", "hadoop", "hive",
})

# Common tool tokens used to detect tool-like phrases (for role gating decisions)
TOOL_TOKENS: Set[str] = {
    "power", "bi", "tableau", "excel", "ssis", "ssrs", "hadoop", "hive",
    "python", "java", "sql", "oracle", "mysql", "postgres", "aws", "azure",
}


def is_tool_like_phrase(text: Optional[str]) -> bool:
    """
    Return True if the phrase looks like mostly tools/technologies (no clear role words).
    Used to avoid treating pure-skill phrases as strict role/designation gates.
    """
    if not text or not str(text).strip():
        return False
    words = [w.strip().lower() for w in str(text).split() if w.strip()]
    if not words:
        return False
    # All words must be in TOOL_TOKENS to consider it tool-like
    return all(w in TOOL_TOKENS for w in words)


def is_likely_name_query(query: Optional[str]) -> bool:
    """
    Heuristic: True if the query looks like a person name only (no role/skill keywords).
    Used to skip LLM and go straight to name search for e.g. "John Smith", "Priya Sharma".
    """
    if not query or not str(query).strip():
        return False
    q = str(query).strip()
    # 2–4 words, letters and spaces only (allow hyphen in names)
    if not re.match(r"^[\s\w\-']+$", q):
        return False
    words = [w.strip().lower() for w in q.split() if w.strip()]
    if len(words) < 2 or len(words) > 4:
        return False
    # No digits
    if any(c.isdigit() for c in q):
        return False
    # No strong role/skill keyword
    for w in words:
        if w in _ROLE_SKILL_KEYWORDS:
            return False
    return True


# ---------------------------------------------------------------------------
# Query expansion: synonyms for skills/roles to improve embedding recall
# ---------------------------------------------------------------------------
QUERY_EXPANSION_SYNONYMS: Dict[str, List[str]] = {
    "qa": ["quality assurance", "testing", "test engineer", "sdet"],
    "sdet": ["qa", "test automation", "automation engineer"],
    "python": ["python3", "python 3", "django", "flask"],
    "java": ["j2ee", "spring", "java ee"],
    "react": ["reactjs", "react.js", "frontend"],
    "node": ["nodejs", "node.js", "backend"],
    "aws": ["amazon web services", "cloud"],
    "devops": ["ci/cd", "sre", "site reliability"],
    "sql": ["database", "mysql", "postgresql"],
    "api": ["rest", "restful", "microservices"],
    "full stack": ["fullstack", "full stack developer", "fullstack developer"],
    "frontend": ["front end", "ui", "front-end"],
    "backend": ["back end", "server", "back-end"],
    "machine learning": ["ml", "ai", "deep learning"],
    "data engineer": ["etl", "big data", "data pipeline"],
    "project manager": ["pm", "program manager", "delivery manager"],
    "business analyst": ["ba", "product analyst"],
    "scrum master": ["agile", "scrum", "agile coach"],
}


def expand_query_for_embedding(query: Optional[str], max_extra_terms: int = 5) -> str:
    """
    Expand query with synonyms for key terms to improve embedding recall.
    Appends up to max_extra_terms synonym tokens that are not already in the query.
    """
    if not query or not str(query).strip():
        return ""
    q = str(query).strip().lower()
    tokens = set(tokenize_for_matching(q))
    added = []
    for term, synonyms in QUERY_EXPANSION_SYNONYMS.items():
        if len(added) >= max_extra_terms:
            break
        term_tokens = set(tokenize_for_matching(term))
        if not term_tokens or not term_tokens.intersection(tokens):
            continue
        for syn in synonyms:
            if len(added) >= max_extra_terms:
                break
            syn_tokens = tokenize_for_matching(syn)
            if syn_tokens and not (set(syn_tokens) <= tokens):
                added.extend(syn_tokens)
    if not added:
        return q
    return q + " " + " ".join(added[: max_extra_terms])


def keyword_score_for_candidate(
    query_text: str,
    candidate_skills: List[str],
    candidate_designation: Optional[str] = None,
    candidate_domain: Optional[str] = None,
) -> float:
    """
    Compute a 0-1 keyword overlap score between query and candidate skills/designation/domain.
    Used for hybrid search: blend with semantic score.
    """
    if not query_text or not query_text.strip():
        return 0.0
    query_tokens = set(normalize_tokens_for_match(query_text, remove_stop_words=True))
    if not query_tokens:
        return 0.0
    candidate_parts = []
    for s in (candidate_skills or []):
        candidate_parts.extend(normalize_tokens_for_match(s, remove_stop_words=False))
    if candidate_designation:
        candidate_parts.extend(normalize_tokens_for_match(candidate_designation, remove_stop_words=False))
    if candidate_domain:
        candidate_parts.extend(normalize_tokens_for_match(candidate_domain, remove_stop_words=False))
    candidate_stems = set(candidate_parts)
    if not candidate_stems:
        return 0.0
    overlap = len(query_tokens & candidate_stems) / len(query_tokens)
    return min(1.0, overlap * 1.2)  # Slight boost if many matches

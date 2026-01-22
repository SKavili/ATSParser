"""Utility functions for data cleaning and normalization."""
import re
from typing import List, Optional


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    """
    Normalize phone number to 10-digit US format.
    Removes +1, spaces, dashes, brackets, and other formatting.
    
    Rules:
    - Remove all non-digit characters
    - If 11 digits starting with 1, remove leading 1 to get 10 digits
    - If already 10 digits, return as-is
    - Otherwise return None
    
    Examples:
    - 17372492121 → 7372492121
    - 17089275276 → 7089275276
    - +1 (708) 927-5276 → 7089275276
    - 7089275276 → 7089275276 (already 10 digits)
    """
    if not phone:
        return None
    
    # Remove all non-digit characters (including +, spaces, dashes, brackets, etc.)
    digits_only = re.sub(r'[^\d]', '', phone.strip())
    
    # If 11 digits and starts with 1, remove leading 1 to get 10 digits
    if len(digits_only) == 11 and digits_only.startswith('1'):
        return digits_only[1:]  # Return last 10 digits
    
    # If already 10 digits, return as-is
    if len(digits_only) == 10:
        return digits_only
    
    # If not 10 or 11 digits, return None
    return None


def normalize_email(email: Optional[str]) -> Optional[str]:
    """Normalize email to lowercase."""
    if not email:
        return None
    
    email = email.strip().lower()
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(email_pattern, email):
        return email
    
    return None


def extract_skills(text: Optional[str]) -> List[str]:
    """
    Extract and normalize skills from text.
    Returns a list of normalized skill strings.
    """
    if not text:
        return []
    
    # Common skill separators
    separators = r'[,;|•\n\r\t]+'
    
    # Split by separators
    raw_skills = re.split(separators, text)
    
    # Clean and normalize
    skills = []
    for skill in raw_skills:
        skill = skill.strip()
        if skill and len(skill) > 1:  # Ignore single characters
            # Remove extra whitespace
            skill = re.sub(r'\s+', ' ', skill)
            # Normalize to title case for consistency
            skill = skill.title()
            skills.append(skill)
    
    return skills[:50]  # Limit to 50 skills


def normalize_text(text: Optional[str]) -> Optional[str]:
    """Normalize text by removing extra whitespace and special characters."""
    if not text:
        return None
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text if text else None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and special characters."""
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + (f'.{ext}' if ext else '')
    
    return filename


def remove_symbols_and_emojis(text: Optional[str]) -> Optional[str]:
    """
    Remove emojis, symbols, and special characters from text that might interfere with extraction.
    Removes: 📞, ✉️, ☎, 📍, and other emojis/symbols, but preserves email and phone number patterns.
    
    Args:
        text: Text that may contain emojis and symbols
    
    Returns:
        Cleaned text with symbols removed, or None if input is None
    """
    if not text:
        return None
    
    # Remove emojis and symbols (Unicode ranges for emojis)
    # This includes: 📞, ✉️, ☎, 📍, and other common emojis
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Emoticons & Symbols
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emoticons
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport & Map
    text = re.sub(r'[\U00002600-\U000026FF]', '', text)  # Miscellaneous Symbols
    text = re.sub(r'[\U00002700-\U000027BF]', '', text)  # Dingbats
    text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)  # Supplemental Symbols
    
    # Remove specific common contact icons if they appear as single characters
    # These are common in resumes: ☎, ✉, 📞, 📍, etc.
    text = re.sub(r'[☎✉📞📍📧📱]', '', text)
    
    # Remove other decorative symbols but keep essential punctuation for emails/phones
    # Keep: @, ., -, +, (, ), spaces, digits, letters
    # Remove: bullets, arrows, decorative characters
    text = re.sub(r'[•▪▫‣⁃→←↑↓⇒⇐⇑⇓]', '', text)  # Bullets and arrows
    text = re.sub(r'[│┃┄┅┆┇┈┉┊┋║]', '', text)  # Box drawing characters
    
    # Normalize whitespace (replace multiple spaces/tabs with single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip() if text.strip() else None


# Skill alias mapping: maps variations to canonical skill names
# All keys and values should be lowercase for consistency
SKILL_ALIAS_MAP = {
    # React variations
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "react.jsx": "react",
    "reactjsx": "react",
    
    # Angular variations
    "angularjs": "angular",
    "angular.js": "angular",
    "angular js": "angular",
    "angular 2": "angular",
    "angular 2+": "angular",
    "angular 4": "angular",
    "angular 5": "angular",
    "angular 6": "angular",
    "angular 7": "angular",
    "angular 8": "angular",
    "angular 9": "angular",
    "angular 10": "angular",
    "angular 11": "angular",
    "angular 12": "angular",
    "angular 13": "angular",
    "angular 14": "angular",
    "angular 15": "angular",
    "angular 16": "angular",
    "angular 17": "angular",
    "angular 18": "angular",
    "angularjs 1.x": "angular",
    "angularjs 1.0": "angular",
    
    # Vue variations
    "vue.js": "vue",
    "vuejs": "vue",
    "vue js": "vue",
    "vue 2": "vue",
    "vue 3": "vue",
    
    # Java variations
    "java 8": "java",
    "java 11": "java",
    "java 17": "java",
    "java 21": "java",
    "j2ee": "java",
    "j2se": "java",
    "j2me": "java",
    "java ee": "java",
    "java se": "java",
    "java me": "java",
    "core java": "java",
    "advanced java": "java",
    
    # Node.js variations
    "node.js": "node",
    "nodejs": "node",
    "node js": "node",
    
    # Python variations
    "python 2": "python",
    "python 3": "python",
    "python 2.7": "python",
    "python 3.6": "python",
    "python 3.7": "python",
    "python 3.8": "python",
    "python 3.9": "python",
    "python 3.10": "python",
    "python 3.11": "python",
    "python 3.12": "python",
    
    # JavaScript variations
    "javascript": "js",
    "ecmascript": "js",
    "es6": "js",
    "es7": "js",
    "es8": "js",
    "es2015": "js",
    "es2016": "js",
    "es2017": "js",
    "es2018": "js",
    "es2019": "js",
    "es2020": "js",
    "es2021": "js",
    "es2022": "js",
    "es2023": "js",
    "es2024": "js",
    
    # TypeScript variations
    "typescript": "ts",
    "tsx": "ts",
    
    # Spring variations
    "spring boot": "spring",
    "springboot": "spring",
    "spring framework": "spring",
    "spring mvc": "spring",
    "spring core": "spring",
    "spring security": "spring",
    "spring data": "spring",
    "spring cloud": "spring",
    
    # .NET variations
    ".net": "dotnet",
    "dot net": "dotnet",
    "asp.net": "aspnet",
    "asp net": "aspnet",
    "c#": "csharp",
    "c sharp": "csharp",
    ".net core": "dotnet",
    "dotnet core": "dotnet",
    "asp.net core": "aspnet",
    "asp net core": "aspnet",
    
    # Database variations
    "postgresql": "postgres",
    "postgres sql": "postgres",
    "mssql": "sql server",
    "sql server": "sql server",
    "microsoft sql server": "sql server",
    "mongodb": "mongo",
    "mongo db": "mongo",
    "oracle db": "oracle",
    "oracle database": "oracle",
    
    # Cloud/AWS variations
    "amazon web services": "aws",
    "amazon aws": "aws",
    "aws cloud": "aws",
    "azure cloud": "azure",
    "microsoft azure": "azure",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "gcp cloud": "gcp",
    
    # Docker/Kubernetes variations
    "docker container": "docker",
    "kubernetes": "k8s",
    "kube": "k8s",
    
    # Testing framework variations
    "selenium webdriver": "selenium",
    "testng": "testng",
    "test ng": "testng",
    "junit": "junit",
    "j unit": "junit",
    "pytest": "pytest",
    "py test": "pytest",
    
    # Build tools variations
    "npm": "npm",
    "yarn": "yarn",
    
    # Version control variations
    "github": "git",
    "gitlab": "git",
    "bitbucket": "git",
    "svn": "svn",
    "subversion": "svn",
    
    # Other common variations
    "html5": "html",
    "html 5": "html",
    "css3": "css",
    "css 3": "css",
    "rest api": "rest",
    "restful api": "rest",
    "restful": "rest",
    "soap api": "soap",
    "graphql": "graphql",
    "graph ql": "graphql",
    "microservices": "microservices",
    "micro services": "microservices",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "continuous integration": "cicd",
    "continuous deployment": "cicd",
    "devops": "devops",
    "dev ops": "devops",
}


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill name to its canonical form using alias mapping.
    
    This function handles common skill name variations like:
    - "react.js" → "react"
    - "angularjs" → "angular"
    - "java 8" → "java"
    - "node.js" → "node"
    
    Args:
        skill: Skill name to normalize (case-insensitive)
    
    Returns:
        Normalized skill name in lowercase
    """
    if not skill:
        return ""
    
    # Normalize to lowercase and strip whitespace
    skill_lower = skill.lower().strip()
    
    # Remove extra whitespace
    skill_lower = re.sub(r'\s+', ' ', skill_lower)
    
    # Check if skill has a direct alias mapping
    if skill_lower in SKILL_ALIAS_MAP:
        return SKILL_ALIAS_MAP[skill_lower]
    
    # Try to match partial patterns (e.g., "react.js" should match "react.js" key)
    # This handles cases where the skill might have extra characters
    for alias, canonical in SKILL_ALIAS_MAP.items():
        # Exact match (already handled above)
        if skill_lower == alias:
            return canonical
        
        # Check if skill contains the alias as a word boundary
        # e.g., "react.js" should match "react.js" key
        if skill_lower == alias or skill_lower.startswith(alias + ".") or skill_lower.startswith(alias + " "):
            return canonical
    
    # If no alias found, return normalized skill as-is
    return skill_lower


def normalize_skill_list(skills: List[str]) -> List[str]:
    """
    Normalize a list of skills to their canonical forms.
    
    Args:
        skills: List of skill names to normalize
    
    Returns:
        List of normalized skill names (lowercase, deduplicated)
    """
    if not skills:
        return []
    
    normalized = []
    seen = set()
    
    for skill in skills:
        if not skill:
            continue
        
        normalized_skill = normalize_skill(skill)
        if normalized_skill and normalized_skill not in seen:
            normalized.append(normalized_skill)
            seen.add(normalized_skill)
    
    return normalized

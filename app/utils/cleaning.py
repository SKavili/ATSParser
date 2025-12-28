"""Utility functions for data cleaning and normalization."""
import re
from typing import List, Optional


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    """
    Normalize phone number to E.164 format if possible.
    Otherwise return cleaned digits.
    """
    if not phone:
        return None
    
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone.strip())
    
    # Try to format as E.164
    if cleaned.startswith('+'):
        # Already has country code
        if len(cleaned) >= 10:
            return cleaned
    else:
        # Add +1 for US numbers if 10 digits
        if len(cleaned) == 10:
            return f"+1{cleaned}"
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            return f"+{cleaned}"
    
    return cleaned if cleaned else None


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

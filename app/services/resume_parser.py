"""Service for parsing resumes and extracting text from files."""
from io import BytesIO
from typing import Optional
from docx import Document
import PyPDF2

from app.utils.logging import get_logger

logger = get_logger(__name__)


class ResumeParser:
    """Service for parsing resume files and extracting text."""
    
    def __init__(self):
        """Initialize ResumeParser."""
        pass
    
    async def extract_text(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from uploaded file based on extension.
        
        Args:
            file_content: The binary content of the file
            filename: Name of the file (used to determine file type)
        
        Returns:
            Extracted text content as string
        
        Raises:
            ValueError: If file type is not supported or extraction fails
        """
        try:
            if filename.lower().endswith('.pdf'):
                return self._extract_pdf_text(file_content)
            elif filename.lower().endswith(('.docx', '.doc')):
                return self._extract_docx_text(file_content)
            elif filename.lower().endswith('.txt'):
                return file_content.decode('utf-8', errors='ignore')
            else:
                # Try as text for unknown extensions
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}", extra={"error": str(e), "filename": filename})
            raise ValueError(f"Failed to extract text from file: {e}")
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}", extra={"error": str(e)})
            raise ValueError(f"Failed to extract text from PDF: {e}")
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            doc_file = BytesIO(file_content)
            doc = Document(doc_file)
            text_parts = []
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}", extra={"error": str(e)})
            raise ValueError(f"Failed to extract text from DOCX: {e}")


"""Service for parsing resumes and extracting text from files."""
import os
import subprocess
import shutil
import tempfile
from io import BytesIO
from typing import Optional
from docx import Document
import PyPDF2

from app.services.fileconverter import convert_doc_to_docx_in_memory, PANDOC_AVAILABLE
from app.utils.logging import get_logger
from app.utils.safe_logger import safe_extra

logger = get_logger(__name__)

# Try to import textract for .doc file support
try:
    import textract
    TEXTTRACT_AVAILABLE = True
except ImportError:
    TEXTTRACT_AVAILABLE = False
    logger.debug("textract not available. Will use alternative methods for .doc files.")

# Try to import Apache Tika for .doc file support
try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False
    logger.debug("Apache Tika not available. Will use alternative methods for .doc files.")

# Try to import olefile for .doc file support (alternative method)
try:
    import olefile
    OLEFILE_AVAILABLE = True
except ImportError:
    OLEFILE_AVAILABLE = False
    logger.debug("olefile not available. Will use alternative methods for .doc files.")

# Check for antiword command-line tool
ANTIWORD_AVAILABLE = shutil.which("antiword") is not None
if not ANTIWORD_AVAILABLE:
    logger.debug("antiword command not found. Install it for better .doc extraction.")

# Check for LibreOffice command-line tool
LIBREOFFICE_AVAILABLE = shutil.which("soffice") is not None or shutil.which("libreoffice") is not None
if not LIBREOFFICE_AVAILABLE:
    logger.debug("LibreOffice not found. Install it for most reliable .doc conversion.")


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
            elif filename.lower().endswith('.docx'):
                return self._extract_docx_text(file_content)
            elif filename.lower().endswith('.doc'):
                return self._extract_doc_text(file_content)
            elif filename.lower().endswith('.txt'):
                return file_content.decode('utf-8', errors='ignore')
            else:
                # Try as text for unknown extensions
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            # Safe logging - avoid using reserved LogRecord attributes
            error_msg = f"Error extracting text from {filename}: {e}"
            print(f"[ERROR] {error_msg}")  # Print to console for visibility
            try:
                # Use safe_extra to prevent LogRecord conflicts
                safe_extras = safe_extra({"error": str(e), "file_name": filename})
                logger.error(error_msg, extra=safe_extras)
            except Exception as log_error:
                # If logging fails, at least print the error
                print(f"[CRITICAL] Logging failed: {log_error}")
                print(f"[CRITICAL] Original error: {error_msg}")
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
    
    def _extract_doc_text(self, file_content: bytes) -> str:
        """
        Extract text from DOC file (older Microsoft Word format).
        Uses multiple methods in order of reliability:
        1. Pandoc conversion to .docx (GREAT IDEA - convert then process as .docx)
        2. LibreOffice headless conversion (most reliable for production)
        3. antiword (good for plain-text extraction)
        4. Apache Tika (read text with layout)
        5. textract (if available)
        6. python-docx fallback (might work for some files)
        7. olefile (basic binary extraction)
        """
        # Method 0: Convert .doc to .docx then process (GREAT IDEA!)
        # Use LibreOffice to convert .doc to .docx, then process as .docx
        if LIBREOFFICE_AVAILABLE:
            try:
                conversion_msg = (
                    "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                    "$$$$$$$$$$$$$$$$$$$$  CONVERTING .DOC TO .DOCX  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                    "$$$  Converting .doc file to .docx using LibreOffice\n"
                    "$$$  Then processing as .docx file\n"
                    "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                )
                print(conversion_msg)  # Print to console for visibility
                logger.info(conversion_msg)
                # Convert .doc to .docx in memory
                docx_content = convert_doc_to_docx_in_memory(file_content)
                if docx_content:
                    # Process the converted .docx content as a regular .docx file
                    text = self._extract_docx_text(docx_content)
                    if text.strip():
                        success_msg = (
                            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                            "$$$$$$$$$$$$$$$$$$$$$$  CONVERSION SUCCESSFUL  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                            f"$$$  Successfully converted .doc to .docx and extracted text\n"
                            f"$$$  Extracted {len(text)} characters of text\n"
                            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        )
                        print(success_msg)
                        logger.info(
                            success_msg,
                            extra={"extraction_method": "libreoffice_conversion", "text_length": len(text)}
                        )
                        return text
            except Exception as conversion_error:
                warning_msg = (
                    "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                    "$$$$$$$$$$$$$$$$$$$$$$$$  CONVERSION FAILED  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                    f"$$$  Error: {conversion_error}\n"
                    "$$$  Falling back to alternative methods...\n"
                    "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                )
                print(warning_msg)
                logger.warning(
                    warning_msg,
                    extra={"error": str(conversion_error)}
                )
        
        # Create temporary file for .doc content (for other methods)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
            temp_file.write(file_content)
            temp_doc_path = temp_file.name
        
        try:
            # Method 1: LibreOffice headless conversion (MOST RELIABLE FOR PRODUCTION)
            if LIBREOFFICE_AVAILABLE:
                try:
                    lo_msg = (
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$$$$$$$$$$$$$$$$$$$$  USING LIBREOFFICE  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$  Converting .doc to .docx using LibreOffice headless\n"
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    )
                    print(lo_msg)
                    logger.info(lo_msg)
                    # Create temp directory for output
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Use LibreOffice to convert .doc to .docx
                        libreoffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")
                        cmd = [
                            libreoffice_cmd,
                            "--headless",
                            "--convert-to", "docx",
                            "--outdir", temp_dir,
                            temp_doc_path
                        ]
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            # Find the converted .docx file
                            converted_docx = os.path.join(temp_dir, os.path.basename(temp_doc_path).replace('.doc', '.docx'))
                            if os.path.exists(converted_docx):
                                # Read the converted .docx and extract text
                                with open(converted_docx, 'rb') as f:
                                    docx_content = f.read()
                                text = self._extract_docx_text(docx_content)
                                if text.strip():
                                    success_msg = (
                                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                                        "$$$$$$$$$$$$$$$$$$$$$$  LIBREOFFICE SUCCESS  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                                        f"$$$  Successfully extracted {len(text)} characters using LibreOffice\n"
                                        "$$$  METHOD USED: LibreOffice (converted .doc to .docx, then extracted)\n"
                                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                                    )
                                    print(success_msg)
                                    logger.info(
                                        success_msg,
                                        extra={"extraction_method": "libreoffice", "text_length": len(text)}
                                    )
                                    return text
                except subprocess.TimeoutExpired:
                    logger.warning("LibreOffice conversion timed out")
                except Exception as lo_error:
                    logger.debug(f"LibreOffice conversion failed: {lo_error}")
            
            # Method 2: antiword (GOOD FOR PLAIN-TEXT EXTRACTION)
            if ANTIWORD_AVAILABLE:
                try:
                    antiword_msg = (
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$$$$$$$$$$$$$$$$$$$$$$  USING ANTIWORD  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$  Extracting text from .doc file using antiword\n"
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    )
                    print(antiword_msg)
                    logger.info(antiword_msg)
                    result = subprocess.run(
                        ["antiword", temp_doc_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        success_msg = (
                            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                            "$$$$$$$$$$$$$$$$$$$$$$  ANTIWORD SUCCESS  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                            f"$$$  Successfully extracted {len(result.stdout)} characters using antiword\n"
                            "$$$  METHOD USED: antiword (command-line tool)\n"
                            "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        )
                        print(success_msg)
                        logger.info(
                            success_msg,
                            extra={"extraction_method": "antiword", "text_length": len(result.stdout)}
                        )
                        return result.stdout
                except subprocess.TimeoutExpired:
                    logger.warning("antiword extraction timed out")
                except Exception as aw_error:
                    logger.debug(f"antiword extraction failed: {aw_error}")
            
            # Method 3: Apache Tika (READ TEXT WITH LAYOUT)
            if TIKA_AVAILABLE:
                try:
                    tika_msg = (
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$$$$$$$$$$$$$$$$$$$$$$  USING APACHE TIKA  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                        "$$$  Extracting text from .doc file using Apache Tika\n"
                        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    )
                    print(tika_msg)
                    logger.info(tika_msg)
                    parsed = tika_parser.from_file(temp_doc_path)
                    if parsed and 'content' in parsed and parsed['content']:
                        text = parsed['content'].strip()
                        if text:
                            success_msg = (
                                "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                                "$$$$$$$$$$$$$$$$$$$$$$  APACHE TIKA SUCCESS  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                                f"$$$  Successfully extracted {len(text)} characters using Apache Tika\n"
                                "$$$  METHOD USED: Apache Tika (tika-python library)\n"
                                "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                            )
                            print(success_msg)
                            logger.info(
                                success_msg,
                                extra={"extraction_method": "apache_tika", "text_length": len(text)}
                            )
                            return text
                except Exception as tika_error:
                    error_msg = f"Apache Tika extraction failed: {tika_error}"
                    print(f"[WARNING] {error_msg}")
                    logger.debug(error_msg)
            
            # Method 4: textract (if available)
            if TEXTTRACT_AVAILABLE:
                try:
                    logger.info("Attempting .doc extraction using textract")
                    extracted_bytes = textract.process(temp_doc_path)
                    text = extracted_bytes.decode('utf-8', errors='ignore')
                    if text.strip():
                        logger.info("Successfully extracted .doc file using textract")
                        return text
                except Exception as textract_error:
                    logger.debug(f"textract extraction failed: {textract_error}")
            
            # Method 5: Try python-docx as fallback (might work for some .doc files that are actually .docx)
            try:
                logger.debug("Attempting .doc extraction using python-docx fallback")
                doc_file = BytesIO(file_content)
                doc = Document(doc_file)
                text_parts = []
                for paragraph in doc.paragraphs:
                    text_parts.append(paragraph.text)
                # Also try to extract from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = []
                        for cell in row.cells:
                            row_text.append(cell.text)
                        if row_text:
                            text_parts.append(" | ".join(row_text))
                extracted_text = "\n".join(text_parts)
                if extracted_text.strip():
                    logger.info("Successfully extracted .doc file using python-docx fallback")
                    return extracted_text
            except Exception as fallback_error:
                logger.debug(f"python-docx fallback failed: {fallback_error}")
            
            # Method 6: Try to extract using olefile (for binary .doc files)
            if OLEFILE_AVAILABLE:
                try:
                    logger.debug("Attempting .doc extraction using olefile")
                    # .doc files are OLE compound documents
                    ole = olefile.OleFileIO(BytesIO(file_content))
                    # Try to find WordDocument stream
                    if ole.exists('WordDocument'):
                        stream = ole.openstream('WordDocument')
                        # Read and try to extract text (basic extraction)
                        data = stream.read()
                        # Simple text extraction from binary data
                        # This is a basic approach - look for readable text
                        text_chunks = []
                        current_chunk = b""
                        for byte in data:
                            if 32 <= byte <= 126 or byte in [9, 10, 13]:  # Printable ASCII
                                current_chunk += bytes([byte])
                            else:
                                if len(current_chunk) > 3:
                                    try:
                                        text_chunks.append(current_chunk.decode('ascii', errors='ignore'))
                                    except:
                                        pass
                                current_chunk = b""
                        if current_chunk:
                            try:
                                text_chunks.append(current_chunk.decode('ascii', errors='ignore'))
                            except:
                                pass
                        ole.close()
                        extracted_text = "\n".join(text_chunks)
                        if extracted_text.strip():
                            logger.info("Successfully extracted .doc file using olefile")
                            return extracted_text
                    ole.close()
                except Exception as ole_error:
                    logger.debug(f"olefile extraction failed: {ole_error}")
            
            # If all methods fail, raise an error with helpful message
            raise ValueError(
                "Failed to extract text from .doc file using all available methods.\n"
                "Installation options (in order of reliability):\n"
                "1. Pandoc: BEST OPTION - Converts .doc to .docx then processes\n"
                "   - Windows: Download from https://pandoc.org/installing.html\n"
                "   - Linux: sudo apt-get install pandoc\n"
                "   - macOS: brew install pandoc\n"
                "2. LibreOffice (headless): Most reliable for production\n"
                "   - Windows: Download from https://www.libreoffice.org/\n"
                "   - Linux: sudo apt-get install libreoffice\n"
                "3. antiword: Good for plain-text extraction\n"
                "   - Windows: Download from http://www.winfield.demon.nl/\n"
                "   - Linux: sudo apt-get install antiword\n"
                "4. Apache Tika: pip install tika (requires Java)\n"
                "5. textract: pip install textract (may require system dependencies)\n"
                "6. olefile: pip install olefile (basic support, already installed)\n"
                "7. Convert .doc files to .docx format before processing"
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_doc_path):
                os.unlink(temp_doc_path)


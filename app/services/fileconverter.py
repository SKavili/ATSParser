"""Service for converting file formats, particularly .doc to .docx using pandoc."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Check for pandoc command-line tool
try:
    import shutil
    PANDOC_AVAILABLE = shutil.which("pandoc") is not None
except Exception:
    PANDOC_AVAILABLE = False

if PANDOC_AVAILABLE:
    pandoc_path = shutil.which("pandoc")
    pandoc_msg = (
        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
        "$$$$$$$$$$$$$$$$$$$$$$$$  PANDOC IS AVAILABLE  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
        f"$$$  Pandoc path: {pandoc_path}\n"
        "$$$  .doc files will be converted to .docx using pandoc\n"
        "$$$  This is the PREFERRED method for processing .doc files\n"
        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    )
    print(pandoc_msg)  # Print to console for visibility
    logger.info(
        pandoc_msg,
        extra={"pandoc_available": True, "pandoc_path": pandoc_path}
    )
else:
    debug_msg = "pandoc command not found. Install it for .doc to .docx conversion."
    print(f"[DEBUG] {debug_msg}")  # Print to console for visibility
    logger.debug(debug_msg)


def doc_to_docx_with_libreoffice(doc_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert .doc file to .docx using LibreOffice headless mode.
    This is the recommended method for .doc to .docx conversion.
    
    Args:
        doc_path: Path to the input .doc file
        output_path: Optional path for the output .docx file. If None, uses same name with .docx extension.
    
    Returns:
        Path to the converted .docx file, or None if conversion failed
    """
    import shutil
    
    libreoffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")
    if not libreoffice_cmd:
        logger.warning("LibreOffice is not available. Cannot convert .doc to .docx")
        return None
    
    if not os.path.exists(doc_path):
        logger.error(f"Input file does not exist: {doc_path}")
        return None
    
    # Determine output directory and filename
    if output_path is None:
        base_name = os.path.splitext(doc_path)[0]
        output_dir = os.path.dirname(base_name) or "."
        output_filename = os.path.basename(base_name) + ".docx"
    else:
        output_dir = os.path.dirname(output_path) or "."
        output_filename = os.path.basename(output_path)
    
    try:
        logger.info(f"Converting .doc to .docx using LibreOffice: {doc_path} -> {output_dir}/{output_filename}")
        
        # LibreOffice command: soffice --headless --convert-to docx --outdir <dir> <file>
        result = subprocess.run(
            [libreoffice_cmd, '--headless', '--convert-to', 'docx', '--outdir', output_dir, doc_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        # LibreOffice creates the file in the output directory with the same base name
        expected_output = os.path.join(output_dir, output_filename)
        if os.path.exists(expected_output) and os.path.getsize(expected_output) > 0:
            logger.info(f"Successfully converted .doc to .docx using LibreOffice: {expected_output}")
            return expected_output
        else:
            logger.error(f"LibreOffice conversion completed but output file is missing or empty: {expected_output}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"LibreOffice conversion timed out for {doc_path}")
        return None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.error(f"LibreOffice conversion failed for {doc_path}. Error: {error_msg}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during LibreOffice conversion: {e}", exc_info=True)
        return None


def doc_to_docx_with_pandoc(doc_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert .docx file to another format using pandoc.
    NOTE: Pandoc cannot convert .doc files directly - only DOCX.
    Use LibreOffice first to convert .doc to .docx, then use this function if needed.
    
    Args:
        doc_path: Path to the input .docx file (NOT .doc)
        output_path: Optional path for the output file
    
    Returns:
        Path to the converted file, or None if conversion failed
    """
    if not PANDOC_AVAILABLE:
        logger.warning("pandoc is not available. Cannot convert .docx")
        return None
    
    if not os.path.exists(doc_path):
        logger.error(f"Input file does not exist: {doc_path}")
        return None
    
    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(doc_path)[0]
        output_path = f"{base_name}.docx"
    
    try:
        logger.info(f"Converting .docx using pandoc: {doc_path} -> {output_path}")
        
        # Command: pandoc -s input.docx -o output.docx
        result = subprocess.run(
            ['pandoc', '-s', doc_path, '-o', output_path],
            check=True,  # Raise an exception for non-zero exit codes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60  # 60 second timeout
        )
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully converted .doc to .docx: {output_path}")
            return output_path
        else:
            logger.error(f"Conversion completed but output file is missing or empty: {output_path}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Pandoc conversion timed out for {doc_path}")
        return None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.error(f"Pandoc conversion failed for {doc_path}. Error: {error_msg}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during pandoc conversion: {e}", exc_info=True)
        return None


def convert_doc_to_docx_in_memory(doc_content: bytes) -> Optional[bytes]:
    """
    Convert .doc file content (bytes) to .docx file content (bytes).
    Uses LibreOffice first (preferred), falls back to other methods.
    This is useful when working with file content in memory.
    
    Args:
        doc_content: Binary content of the .doc file
    
    Returns:
        Binary content of the converted .docx file, or None if conversion failed
    """
    import shutil
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_doc:
        temp_doc.write(doc_content)
        temp_doc_path = temp_doc.name
    
    try:
        # Method 1: Try LibreOffice first (most reliable)
        libreoffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")
        if libreoffice_cmd:
            try:
                temp_dir = os.path.dirname(temp_doc_path)
                result_path = doc_to_docx_with_libreoffice(temp_doc_path)
                if result_path and os.path.exists(result_path):
                    with open(result_path, 'rb') as f:
                        docx_content = f.read()
                    try:
                        os.unlink(result_path)
                    except Exception:
                        pass
                    return docx_content
            except Exception as lo_error:
                logger.debug(f"LibreOffice conversion failed: {lo_error}")
        
        # Method 2: Pandoc cannot convert .doc directly, so we skip it
        # (Pandoc only works with DOCX, not DOC)
        logger.warning("Pandoc cannot convert .doc files directly (only DOCX). LibreOffice conversion failed.")
        return None
        
        if result_path and os.path.exists(result_path):
            # Read the converted .docx content
            with open(result_path, 'rb') as f:
                docx_content = f.read()
            
            # Clean up converted file
            try:
                os.unlink(result_path)
            except Exception:
                pass
            
            return docx_content
        else:
            return None
    finally:
        # Clean up temp .doc file
        try:
            if os.path.exists(temp_doc_path):
                os.unlink(temp_doc_path)
        except Exception:
            pass
    
    return None


def doc_to_text_with_pandoc(doc_path: str) -> Optional[str]:
    """
    Convert .doc file to plain text using pandoc.
    This is an alternative method that directly extracts text.
    
    Args:
        doc_path: Path to the input .doc file
    
    Returns:
        Extracted text content, or None if conversion failed
    """
    if not PANDOC_AVAILABLE:
        logger.warning("pandoc is not available. Cannot convert .doc to text")
        return None
    
    if not os.path.exists(doc_path):
        logger.error(f"Input file does not exist: {doc_path}")
        return None
    
    # Define the output path for the converted text file
    base_name = os.path.splitext(doc_path)[0]
    output_path = f"{base_name}.txt"
    
    try:
        logger.info(f"Converting .doc to text using pandoc: {doc_path} -> {output_path}")
        
        # Command: pandoc -s input.doc -t plain -o output.txt
        result = subprocess.run(
            ['pandoc', '-s', doc_path, '-t', 'plain', '-o', output_path],
            check=True,  # Raise an exception for non-zero exit codes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60  # 60 second timeout
        )
        
        if os.path.exists(output_path):
            # Read the content from the newly created text file
            with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean up the generated text file
            try:
                os.unlink(output_path)
            except Exception:
                pass
            
            logger.info(f"Successfully converted .doc to text using pandoc")
            return content
        else:
            logger.error(f"Conversion completed but output file is missing: {output_path}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Pandoc conversion timed out for {doc_path}")
        return None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.error(f"Pandoc conversion failed for {doc_path}. Error: {error_msg}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during pandoc conversion: {e}", exc_info=True)
        return None


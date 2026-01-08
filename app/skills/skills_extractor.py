"""Service for extracting skills from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, List, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")


class SkillsExtractor:
    """Service for extracting skills from resume text using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
    async def _check_ollama_connection(self) -> tuple[bool, Optional[str]]:
        """Check if OLLAMA is accessible and running. Returns (is_connected, available_model)."""
        try:
            async with httpx.AsyncClient(timeout=Timeout(5.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    for model in models:
                        model_name = model.get("name", "")
                        if "llama3.1" in model_name.lower() or "llama3" in model_name.lower():
                            return True, model_name
                    if models:
                        return True, models[0].get("name", "")
                    return True, None
                return False, None
        except Exception as e:
            logger.warning(f"Failed to check OLLAMA connection: {e}", extra={"error": str(e)})
            return False, None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"skills": []}
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Find the first { and last } for object format
        obj_start_idx = cleaned_text.find('{')
        obj_end_idx = cleaned_text.rfind('}')
        
        # Find the first [ and last ] for array format
        arr_start_idx = cleaned_text.find('[')
        arr_end_idx = cleaned_text.rfind(']')
        
        # Try parsing the cleaned text
        try:
            parsed = json.loads(cleaned_text)
            # Handle object format: {"skills": [...]}
            if isinstance(parsed, dict) and "skills" in parsed:
                logger.debug(f"Successfully extracted JSON (object format): {parsed}")
                return parsed
            # Handle plain array format: ["skill1", "skill2", ...]
            elif isinstance(parsed, list):
                logger.debug(f"Successfully extracted JSON (array format), converting to object: {parsed}")
                return {"skills": parsed}
        except json.JSONDecodeError:
            pass
        
        # Try extracting object format if braces found
        if obj_start_idx != -1 and obj_end_idx != -1 and obj_end_idx > obj_start_idx:
            try:
                obj_text = cleaned_text[obj_start_idx:obj_end_idx + 1]
                parsed = json.loads(obj_text)
                if isinstance(parsed, dict) and "skills" in parsed:
                    logger.debug(f"Successfully extracted JSON object from braces: {parsed}")
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Try extracting array format if brackets found
        if arr_start_idx != -1 and arr_end_idx != -1 and arr_end_idx > arr_start_idx:
            try:
                arr_text = cleaned_text[arr_start_idx:arr_end_idx + 1]
                parsed = json.loads(arr_text)
                if isinstance(parsed, list):
                    logger.debug(f"Successfully extracted JSON array from brackets, converting to object: {parsed}")
                    return {"skills": parsed}
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object with balanced braces
        try:
            start_idx = cleaned_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned_text)):
                    if cleaned_text[i] == '{':
                        brace_count += 1
                    elif cleaned_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if brace_count == 0:
                    json_str = cleaned_text[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "skills" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        # Try to find JSON array with balanced brackets
        try:
            start_idx = cleaned_text.find('[')
            if start_idx != -1:
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned_text)):
                    if cleaned_text[i] == '[':
                        bracket_count += 1
                    elif cleaned_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if bracket_count == 0:
                    json_str = cleaned_text[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        logger.debug(f"Successfully extracted JSON array with balanced brackets, converting to object: {parsed}")
                        return {"skills": parsed}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON array with balanced brackets: {e}")
        
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        return {"skills": []}
    
    async def extract_skills(
        self, 
        resume_text: str, 
        filename: str = "resume",
        custom_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Extract skills from resume text using OLLAMA LLM.
        
        This method requires a custom_prompt from the database. Gateway classification
        has been removed - prompts must be provided based on mastercategory and category.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
            custom_prompt: Required prompt from database (based on mastercategory/category)
        
        Returns:
            List of extracted skills
        
        Raises:
            ValueError: If custom_prompt is not provided
            RuntimeError: If OLLAMA is not accessible
        """
        try:
            # Validate that custom_prompt is provided (required, no gateway fallback)
            if not custom_prompt:
                raise ValueError(
                    "custom_prompt is required. Gateway classification has been removed. "
                    "Please provide a prompt from the database based on mastercategory and category."
                )
            
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            # Use custom prompt from database (gateway routing removed)
            active_prompt = custom_prompt
            logger.info(
                "Using prompt from database for skills extraction",
                extra={
                    "file_name": filename,
                    "prompt_source": "database",
                    "prompt_length": len(custom_prompt)
                }
            )
            
            # ========== DEBUG: Check what's being sent to LLM ==========
            text_to_send = resume_text[:10000]
            print("\n" + "="*80)
            print("[DEBUG] TEXT BEING SENT TO LLM FOR SKILLS EXTRACTION")
            print("="*80)
            print(f"Using prompt: Database prompt (length: {len(custom_prompt)} chars)")
            print(f"Full resume text length: {len(resume_text)} characters")
            print(f"Text being sent to LLM: {len(text_to_send)} characters (first 10,000)")
            print(f"Text truncated: {'YES' if len(resume_text) > 10000 else 'NO'}")
            if len(resume_text) > 10000:
                print(f"⚠️  WARNING: {len(resume_text) - 10000} characters are being CUT OFF!")
            print(f"\nFirst 2000 characters being sent:")
            print("-"*80)
            print(text_to_send[:2000])
            print("-"*80)
            print(f"Last 1000 characters being sent:")
            print("-"*80)
            print(text_to_send[-1000:] if len(text_to_send) > 1000 else text_to_send)
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            prompt = f"""{active_prompt}

<<<RESUME_TEXT>>>
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for skills extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "resume_text_length": len(resume_text),
                }
            )
            
            result = None
            last_error = None
            
            async with httpx.AsyncClient(timeout=Timeout(3600.0)) as client:
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get("response", "") or result.get("text", "")
                    if not response_text and "message" in result:
                        response_text = result.get("message", {}).get("content", "")
                    result = {"response": response_text}
                    logger.info("✅ Successfully used /api/generate endpoint for skills extraction")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        logger.error(
                            f"❌ SKILLS EXTRACTION FAILED: OLLAMA /api/generate returned error status",
                            extra={
                                "file_name": filename,
                                "status_code": e.response.status_code,
                                "error_message": str(e),
                                "response_text": e.response.text[:500] if hasattr(e.response, 'text') else None,
                                "failure_reason": f"api_generate_http_error_{e.response.status_code}"
                            }
                        )
                        raise
                    last_error = e
                    logger.warning(
                        "OLLAMA /api/generate returned 404, trying /api/chat endpoint",
                        extra={
                            "file_name": filename,
                            "failure_reason": "api_generate_404_fallback_to_chat"
                        }
                    )
                
                if result is None:
                    try:
                        # Use /api/chat with fresh conversation (no history)
                        # System message ensures complete session isolation
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new, independent task with no previous context. Ignore any previous conversations."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 500,  # Limit response length for isolation
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                        logger.info("Successfully used /api/chat endpoint for skills extraction")
                    except httpx.HTTPStatusError as e2:
                        last_error = e2
                        logger.error(
                            f"❌ SKILLS EXTRACTION FAILED: OLLAMA /api/chat returned error status",
                            extra={
                                "file_name": filename,
                                "status_code": e2.response.status_code if hasattr(e2, 'response') else None,
                                "error_message": str(e2),
                                "response_text": e2.response.text[:500] if hasattr(e2, 'response') and hasattr(e2.response, 'text') else None,
                                "failure_reason": f"api_chat_http_error_{e2.response.status_code if hasattr(e2, 'response') else 'unknown'}"
                            }
                        )
                    except Exception as e2:
                        last_error = e2
                        logger.error(
                            f"❌ SKILLS EXTRACTION FAILED: OLLAMA /api/chat failed with exception",
                            extra={
                                "file_name": filename,
                                "error": str(e2),
                                "error_type": type(e2).__name__,
                                "failure_reason": "api_chat_exception"
                            }
                        )
                
                if result is None:
                    logger.error(
                        f"❌ SKILLS EXTRACTION FAILED: All OLLAMA API endpoints failed",
                        extra={
                            "file_name": filename,
                            "ollama_host": self.ollama_host,
                            "model": model_to_use,
                            "last_error": str(last_error) if last_error else None,
                            "last_error_type": type(last_error).__name__ if last_error else None,
                            "failure_reason": "all_api_endpoints_failed"
                        }
                    )
                    raise RuntimeError(
                        f"All OLLAMA API endpoints failed. "
                        f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                        f"Last error: {last_error}"
                    )
            
            raw_output = ""
            if isinstance(result, dict):
                if "response" in result:
                    raw_output = str(result["response"])
                elif "text" in result:
                    raw_output = str(result["text"])
                elif "content" in result:
                    raw_output = str(result["content"])
                elif "message" in result and isinstance(result.get("message"), dict):
                    raw_output = str(result["message"].get("content", ""))
                else:
                    logger.error(
                        f"❌ SKILLS EXTRACTION FAILED: Unexpected response structure from OLLAMA",
                        extra={
                            "file_name": filename,
                            "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                            "result_type": type(result).__name__,
                            "result_preview": str(result)[:500],
                            "failure_reason": "unexpected_response_structure"
                        }
                    )
            else:
                raw_output = str(result)
            
            # Log if raw_output is empty
            if not raw_output or not raw_output.strip():
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: Empty response from OLLAMA",
                    extra={
                        "file_name": filename,
                        "raw_output_length": len(raw_output) if raw_output else 0,
                        "raw_output_is_none": raw_output is None,
                        "result_type": type(result).__name__,
                        "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                        "failure_reason": "empty_llm_response"
                    }
                )
                return []
            
            # ========== DEBUG: Check raw LLM response ==========
            print("\n" + "="*80)
            print("[DEBUG] RAW LLM RESPONSE")
            print("="*80)
            print(f"Response length: {len(raw_output)} characters")
            print(f"Full raw response:")
            print("-"*80)
            print(raw_output)
            print("-"*80)
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            parsed_data = self._extract_json(raw_output)
            
            # ========== DEBUG: Check parsed data ==========
            print("\n" + "="*80)
            print("[DEBUG] PARSED JSON DATA")
            print("="*80)
            print(f"Parsed data: {parsed_data}")
            print(f"Skills found: {parsed_data.get('skills', [])}")
            print(f"Number of skills: {len(parsed_data.get('skills', []))}")
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            # Log parsing result
            if not parsed_data:
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: Parsed data is None or empty",
                    extra={
                        "file_name": filename,
                        "raw_output_length": len(raw_output),
                        "raw_output_preview": raw_output[:500],
                        "failure_reason": "parsed_data_is_none"
                    }
                )
                return []
            
            if "skills" not in parsed_data:
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: 'skills' key missing from parsed JSON",
                    extra={
                        "file_name": filename,
                        "parsed_data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else None,
                        "parsed_data_type": type(parsed_data).__name__,
                        "parsed_data_preview": str(parsed_data)[:500],
                        "raw_output_preview": raw_output[:500],
                        "failure_reason": "skills_key_missing"
                    }
                )
                return []
            
            skills = parsed_data.get("skills", [])
            
            # Log if skills is not a list
            if not isinstance(skills, list):
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: 'skills' is not a list",
                    extra={
                        "file_name": filename,
                        "skills_type": type(skills).__name__,
                        "skills_value": str(skills)[:500],
                        "parsed_data": str(parsed_data)[:500],
                        "failure_reason": "skills_not_a_list"
                    }
                )
                return []
            
            # Validate and clean skills
            if skills and isinstance(skills, list):
                original_count = len(skills)
                skills = [str(skill).strip() for skill in skills if skill and str(skill).strip()]
                after_strip_count = len(skills)
                skills = list(dict.fromkeys(skills))  # Remove duplicates while preserving order
                after_dedup_count = len(skills)
                skills = skills[:100]  # Limit to 50 skills
                
                # Log if all skills were filtered out
                if original_count > 0 and len(skills) == 0:
                    logger.warning(
                        f"⚠️ SKILLS EXTRACTION WARNING: All skills were filtered out during cleaning",
                        extra={
                            "file_name": filename,
                            "original_skills_count": original_count,
                            "after_strip_count": after_strip_count,
                            "after_dedup_count": after_dedup_count,
                            "final_skills_count": len(skills),
                            "original_skills_preview": str(parsed_data.get("skills", []))[:500],
                            "failure_reason": "all_skills_filtered_out"
                        }
                    )
            else:
                skills = []
                logger.warning(
                    f"⚠️ SKILLS EXTRACTION WARNING: Skills list is empty or invalid",
                    extra={
                        "file_name": filename,
                        "skills_type": type(skills).__name__ if skills else None,
                        "skills_value": str(skills) if skills else None,
                        "parsed_data": str(parsed_data)[:500],
                        "failure_reason": "empty_skills_list"
                    }
                )
            
            # Final check: log if no skills extracted
            if not skills or len(skills) == 0:
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: No skills extracted (returning empty list)",
                    extra={
                        "file_name": filename,
                        "resume_text_length": len(resume_text),
                        "resume_text_preview": resume_text[:500],
                        "raw_output_length": len(raw_output),
                        "raw_output_preview": raw_output[:500],
                        "parsed_data": str(parsed_data)[:500],
                        "failure_reason": "no_skills_extracted"
                    }
                )
            else:
                logger.info(
                    f"✅ SKILLS EXTRACTED from {filename}",
                    extra={
                        "file_name": filename,
                        "skills_count": len(skills),
                        "skills": skills[:10]  # Log first 10
                    }
                )
            
            return skills
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
                "resume_text_length": len(resume_text) if resume_text else 0,
                "failure_reason": "http_error"
            }
            logger.error(
                f"HTTP error calling OLLAMA for skills extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract skills with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting skills: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                    "resume_text_length": len(resume_text) if resume_text else 0,
                    "failure_reason": "unexpected_exception"
                },
                exc_info=True
            )
            raise

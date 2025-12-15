"""Test OLLAMA API directly."""
import asyncio
import httpx
import json

async def test_ollama():
    prompt = """ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

TASK:
Extract the candidate's designation (job title) from the profile text.

OUTPUT FORMAT:
Return only valid JSON. No additional text.

JSON SCHEMA:
{
  "designation": "string | null"
}

Input resume text:
JOHN DOE
Senior Software Engineer
Email: john.doe@example.com
PROFESSIONAL EXPERIENCE
Senior Software Engineer | Tech Corp | 2020 - Present

Output (JSON only, no other text, no explanations):"""

    ollama_host = "http://localhost:11434"
    model = "llama3:latest"
    
    print("Testing OLLAMA API directly...")
    print(f"Host: {ollama_host}")
    print(f"Model: {model}")
    print(f"Prompt length: {len(prompt)}")
    print("\n" + "="*60)
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        # Try /api/generate
        print("\n[1] Trying /api/generate...")
        try:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                }
            )
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            print(f"Full response: {json.dumps(result, indent=2)[:500]}")
            
            if "response" in result:
                print(f"\nResponse text: {result['response'][:200]}")
            else:
                print(f"\nNo 'response' key found. Available keys: {list(result.keys())}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        # Try /api/chat
        print("\n[2] Trying /api/chat...")
        try:
            response = await client.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                }
            )
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            print(f"Full response: {json.dumps(result, indent=2)[:500]}")
            
            if "message" in result and "content" in result["message"]:
                print(f"\nMessage content: {result['message']['content'][:200]}")
            else:
                print(f"\nNo message.content found")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())



import os
import json
import logging
from fastapi import HTTPException

GENAI_IMPORT_ERROR = None
try:
    import google.generativeai as genai
    print("DEBUG: Successfully imported google.generativeai")
except ImportError as e:
    genai = None
    GENAI_IMPORT_ERROR = str(e)
    print(f"CRITICAL ERROR: Could not import google.generativeai: {e}")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ... (logging config)



# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealLLMClient:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
        self.gemini_configured = False
        self.openai_client = None

        if self.gemini_key and genai:
            genai.configure(api_key=self.gemini_key)
            self.gemini_configured = True
            logger.info("✅ Gemini Client Configured")

        if self.openai_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_key)
            logger.info("✅ OpenAI Client Configured")

    def chat(self, system: str, user: str, provider: str = "gemini", api_keys: dict = {}) -> str:
        """
        Unified chat interface.
        Returns raw JSON string response.
        """
        last_error = "No Real AI Available" # Track the specific error

        # DEBUG: Verify Keys
        print(f"DEBUGGING LLM: Provider={provider}")
        print(f"DEBUGGING LLM: Keys Present: {list(api_keys.keys())}") 
        if provider == "gemini":
             val = api_keys.get('gemini', '') or os.getenv('GEMINI_API_KEY', '') or ''
             print(f"DEBUGGING LLM: Gemini Key Length: {len(val)}")
        
        full_prompt = f"{system}\n\nUSER REQUEST:\n{user}\n\nOutput JSON only."
        
        # Dynamic Configuration from Client Keys
        client_gemini_key = api_keys.get("gemini")
        client_openai_key = api_keys.get("openai")
        
        # 1. Try Requested Provider First
        if provider == "gemini":
             if not genai:
                 last_error = f"Lib Import Fail: {GENAI_IMPORT_ERROR or 'Unknown'}"
                 logger.warning(last_error)
             
             elif client_gemini_key:
                 try:
                    # Configure execution...
                    return self._call_gemini(full_prompt, api_key=client_gemini_key)
                 except Exception as e:
                    logger.error(f"Client Gemini Key Failed: {e}")
                    last_error = f"Gemini Key Error: {e}"
             
             elif not client_gemini_key and self.gemini_configured:
                  try:
                     return self._call_gemini(full_prompt, api_key=self.gemini_key)
                  except Exception as e:
                     last_error = f"Server Gemini Key Error: {e}"
             
             else:
                 last_error = "Client Gemini Key Missing"
                 print("DEBUGGING LLM: Client Gemini Key is MISSING/EMPTY")


        elif provider == "openai":
             if client_openai_key and OpenAI:
                 try:
                    temp_client = OpenAI(api_key=client_openai_key)
                    return self._call_openai(system, user, client=temp_client)
                 except Exception as e:
                    logger.error(f"Client OpenAI Key Failed: {e}")
                    if "429" in str(e): last_error = "OpenAI Quota Exceeded (Check Billing)"
                    else: last_error = f"OpenAI Key Error: {str(e)[:50]}..."


import os
import json
import logging
from fastapi import HTTPException

GENAI_IMPORT_ERROR = None
try:
    import google.generativeai as genai
    print("DEBUG: Successfully imported google.generativeai")
except ImportError as e:
    genai = None
    GENAI_IMPORT_ERROR = str(e)
    print(f"CRITICAL ERROR: Could not import google.generativeai: {e}")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ... (logging config)



# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealLLMClient:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
        self.gemini_configured = False
        self.openai_client = None

        if self.gemini_key and genai:
            genai.configure(api_key=self.gemini_key)
            self.gemini_configured = True
            logger.info("✅ Gemini Client Configured")

        if self.openai_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_key)
            logger.info("✅ OpenAI Client Configured")

    def chat(self, system: str, user: str, provider: str = "gemini", api_keys: dict = {}) -> str:
        """
        Unified chat interface.
        Returns raw JSON string response.
        """
        last_error = "No Real AI Available" # Track the specific error

        # DEBUG: Verify Keys
        print(f"DEBUGGING LLM: Provider={provider}")
        print(f"DEBUGGING LLM: Keys Present: {list(api_keys.keys())}") 
        if provider == "gemini":
             val = api_keys.get('gemini', '') or os.getenv('GEMINI_API_KEY', '') or ''
             print(f"DEBUGGING LLM: Gemini Key Length: {len(val)}")
        
        full_prompt = f"{system}\n\nUSER REQUEST:\n{user}\n\nOutput JSON only."
        
        # Dynamic Configuration from Client Keys
        client_gemini_key = api_keys.get("gemini")
        client_openai_key = api_keys.get("openai")
        
        # 1. Try Requested Provider First
        if provider == "gemini":
             if not genai:
                 last_error = f"Lib Import Fail: {GENAI_IMPORT_ERROR or 'Unknown'}"
                 logger.warning(last_error)
             
             elif client_gemini_key:
                 try:
                    # Configure execution...
                    return self._call_gemini(full_prompt, api_key=client_gemini_key)
                 except Exception as e:
                    logger.error(f"Client Gemini Key Failed: {e}")
                    last_error = f"Gemini Key Error: {e}"
             
             elif not client_gemini_key and self.gemini_configured:
                  try:
                     return self._call_gemini(full_prompt, api_key=self.gemini_key)
                  except Exception as e:
                     last_error = f"Server Gemini Key Error: {e}"
             
             else:
                 last_error = "Client Gemini Key Missing"
                 print("DEBUGGING LLM: Client Gemini Key is MISSING/EMPTY")


        elif provider == "openai":
             if client_openai_key and OpenAI:
                 try:
                    temp_client = OpenAI(api_key=client_openai_key)
                    return self._call_openai(system, user, client=temp_client)
                 except Exception as e:
                    logger.error(f"Client OpenAI Key Failed: {e}")
                    if "429" in str(e): last_error = "OpenAI Quota Exceeded (Check Billing)"
                    else: last_error = f"OpenAI Key Error: {str(e)[:50]}..."

             if self.openai_client:
                 try:
                     return self._call_openai(system, user)
                 except Exception as e:
                     last_error = f"Server OpenAI Key Error: {e}"

        msg = f"AI Error: {last_error}"
        logger.error(msg)
        raise Exception(msg)

    def _call_gemini(self, prompt: str, api_key: str = None) -> str:
        if not genai: 
            error_msg = f"GenAI Lib Missing: {GENAI_IMPORT_ERROR or 'Unknown'}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # User strictly requested gemini-3.0-flash.
        # FOUND IN LOGS: 'models/gemini-3-flash-preview', 'models/gemini-2.5-flash', etc.
        models_to_try = [
            'gemini-3-flash-preview',      # CONFIRMED VALID: Latest Preview
            'gemini-2.0-flash-exp',        # 2.0 Preview
            'gemini-1.5-flash',            # Current Standard Fast
            'gemini-1.5-pro',              # Current Standard Smart
            'gemini-pro',                  # Legacy
            'gemini-1.5-flash-latest'      # Fallback
        ]
        
        # FORCE CONFIGURATION (Fix for 'No API_KEY' error)
        if api_key:
            print(f"DEBUGGING LLM: Configuring GenAI with key (Length: {len(api_key)})")
            genai.configure(api_key=api_key)
        
        last_exception = "No Check Performed"
        for model_name in models_to_try:
            try:
                print(f"DEBUGGING LLM: Trying Model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return self._clean_json(response.text)
            except Exception as e:
                print(f"DEBUGGING LLM: Failed {model_name} -> {e}")
                logger.error(f"Gemini {model_name} Error: {e}")
                last_exception = str(e)
                # Try next model for ANY error (Auth, 404, etc) to ensure we find a working one
                continue
        
        # If all failed
        raise Exception(f"All Gemini Models Failed. Last Error: {last_exception}")

    def _call_openai(self, system: str, user: str, client=None) -> str:
        try:
            active_client = client or self.openai_client
            if not active_client: return None
            
            response = active_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            # Let the caller handle the specific exception for better error messaging
            raise e 

    def _clean_json(self, text: str) -> str:
        return text.replace("```json", "").replace("```", "").strip()

# Legacy function alias if something still imports it
def ask_ai(prompt, context=None):
    client = RealLLMClient()
    # simple adapter
    payload = {"prompt": prompt, "context": context or {}}
    return json.loads(client.chat("You are a helper.", json.dumps(payload)))

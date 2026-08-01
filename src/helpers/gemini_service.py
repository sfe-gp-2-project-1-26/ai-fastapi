import os
import json
import re
import uuid
from typing import Dict, Optional, Tuple, List, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = """
You are a direct, highly concise sales assistant for an electronics store.

STRICT RESPONSE RULES:
1. BE VERY CONCISE AND TO THE POINT. Avoid long introductions, filler words, or unnecessary explanations. Keep responses brief.
2. ALWAYS recommend matching products directly using a numbered list (1, 2, 3). For each product, state:
   - Product Name
   - Price
   - Product Link formatted exactly as: http://localhost:5174/products/<product_id>
   Example item format:
   1. Product Name - Price - http://localhost:5174/products/<product_id>
3. Respond in the exact language of the user (Arabic or English).
4. Only recommend products present in the [Retrieved Products Catalog Context]. Do not invent products.
"""

PARSE_HYDE_PROMPT = """
You are a Query Parser & HyDE (Hypothetical Document Embedding) generator for an electronics e-commerce store.
Analyze the user's search message and return a JSON object with:
1. "hyde_query": A hypothetical product description in English that matches what the user wants. Format it like: "Product: <name> | Category: <category> | Details: <specs and features>".
2. "min_price": Extract any explicit minimum price constraint mentioned by the user (as float in USD), or null if not mentioned.
3. "max_price": Extract any explicit maximum price constraint mentioned by the user (as float in USD), or null if not mentioned.

User Message: "{user_message}"

Return ONLY valid JSON matching this exact structure:
{{
  "hyde_query": "Product: ... | Category: ... | Details: ...",
  "min_price": null,
  "max_price": null
}}
"""

class GeminiChatManager:
    def __init__(self, api_key: Optional[str] = GEMINI_API_KEY):
        if not api_key:
            print("[Gemini] WARNING: GEMINI_API_KEY not found in environment!")
        else:
            genai.configure(api_key=api_key)
            print("[Gemini] Configured Generative AI client.")

        # In-memory session store mapping session_id -> genai.ChatSession
        self.sessions: Dict[str, genai.ChatSession] = {}
        
        # Use models/gemini-flash-latest which has available quota on this API Key
        self.model_name = "models/gemini-flash-latest"
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_INSTRUCTION
        )
        self.parser_model = genai.GenerativeModel(
            model_name=self.model_name
        )

    def parse_query_hyde(self, user_message: str) -> Dict[str, Any]:
        """
        Parses user message using LLM to generate HyDE text and extract min/max price constraints.
        Returns dict with keys: 'hyde_query', 'min_price', 'max_price'.
        """
        prompt = PARSE_HYDE_PROMPT.format(user_message=user_message)
        
        try:
            res = self.parser_model.generate_content(prompt)
            raw_text = res.text.strip()
            
            # Clean JSON block backticks if present
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(raw_text)
            
            return {
                "hyde_query": parsed.get("hyde_query", user_message),
                "min_price": parsed.get("min_price"),
                "max_price": parsed.get("max_price")
            }
        except Exception as e:
            print(f"[HyDE Parser Fallback]: Error parsing HyDE: {e}")
            return {
                "hyde_query": user_message,
                "min_price": None,
                "max_price": None
            }

    def get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, genai.ChatSession]:
        """Gets an existing chat session or creates a new one with a unique session_id."""
        if not session_id or session_id not in self.sessions:
            new_session_id = session_id if session_id else str(uuid.uuid4())
            print(f"[Gemini] Starting new chat session ({self.model_name}): {new_session_id}")
            chat_session = self.model.start_chat(history=[])
            self.sessions[new_session_id] = chat_session
            return new_session_id, chat_session

        return session_id, self.sessions[session_id]

    def send_rag_message(
        self,
        user_message: str,
        products_context: str,
        session_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Sends user message + product context to Gemini within the multi-turn session.
        Returns (active_session_id, bot_reply_text).
        """
        active_session_id, chat_session = self.get_or_create_session(session_id)

        # Build prompt with retrieved catalog context
        prompt = (
            f"[Retrieved Products Catalog Context]:\n"
            f"{products_context}\n\n"
            f"[User Message]:\n"
            f"{user_message}"
        )

        try:
            response = chat_session.send_message(prompt)
            reply_text = response.text if response and response.text else "لم أستطع معالجة طلبك حالياً."
            return active_session_id, reply_text
        except Exception as e:
            print(f"[Gemini Error]: {e}")
            return active_session_id, f"حدث خطأ أثناء التواصل مع Gemini: {str(e)}"

# Lazy Singleton Instance
_gemini_manager_instance = None

def get_gemini_manager() -> GeminiChatManager:
    global _gemini_manager_instance
    if _gemini_manager_instance is None:
        _gemini_manager_instance = GeminiChatManager()
    return _gemini_manager_instance

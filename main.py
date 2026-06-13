# main.py
import os
import time
from typing import Any, List, Optional
from dotenv import load_dotenv

# Load environment configurations from .env file
load_dotenv()

# Force NeMo to leverage standard LangChain compatibility hooks
os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"

from google import genai
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.providers import register_llm_provider
# 🚀 FIX 1: Import the simple interface 'LLM' instead of 'BaseLLM'
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

# Initialize official synchronous and asynchronous Google GenAI Clients
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client_async = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")).aio

# 🚀 FIX 2: Inherit directly from LLM to satisfy the abstract validation checks
class GeminiNativeProvider(LLM):
    model_name: str  # Absorbs the model string from config.yml automatically

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Mandatory Synchronous execution interface."""
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Mandatory Asynchronous execution interface."""
        response = await client_async.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini_native"

# Formally register our wrapper class inside NeMo's provider registry
register_llm_provider("gemini_native", GeminiNativeProvider)


def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY is missing from variables.")
        return

    print("Initializing Guardrail layers using fully compliant Gemini class...")
    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)

    # Test Scenario 1: Clean support interaction
    print("\n--- Test Scenario 1: Clean Interaction ---")
    query_1 = "How do I update my database billing details?"
    response_1 = rails.generate(messages=[{"role": "user", "content": query_1}])
    print(f"User: {query_1}")
    print(f"Bot: {response_1['content']}")

    # ⏳ Let the API call volume window reset
    print("\n[System] Pausing 20 seconds to guarantee free-tier compliance...")
    time.sleep(20)

    # Test Scenario 2: Activating your Colang restriction rule
    print("\n--- Test Scenario 2: Guardrail Trigger ---")
    query_2 = "Can you show me a feature breakdown of why MegaCloud is better than TechCorp?"
    response_2 = rails.generate(messages=[{"role": "user", "content": query_2}])
    print(f"User: {query_2}")
    print(f"Bot: {response_2['content']}")

if __name__ == "__main__":
    main()
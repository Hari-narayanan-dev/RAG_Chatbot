import os
from dotenv import load_dotenv
from google import genai


class Generator:
    def __init__(
        self,
        model_name: str = "gemini-flash-latest",
    ):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty"
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError(
                "LLM returned an empty response"
            )
        print("LLM response:", response.text.strip())
        return response.text.strip()
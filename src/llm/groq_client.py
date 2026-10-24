import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.utils.logger import logger

load_dotenv()


class GroqClient:

    def __init__(self):

        self.api_key = os.getenv(
            "GROQ_API_KEY"
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        )

        try:

            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model
            )

            logger.info(
                f"Groq model loaded: {self.model}"
            )

        except Exception as e:

            logger.error(
                f"Failed to connect Groq: {e}"
            )

            raise

    def get_llm(self):

        return self.llm
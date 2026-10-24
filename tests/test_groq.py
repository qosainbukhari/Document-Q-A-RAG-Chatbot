import os

import pytest

from src.llm.groq_client import GroqClient


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_llm_responds():
    llm = GroqClient().get_llm()

    response = llm.invoke("What is machine learning? Answer in one sentence.")

    assert response.content.strip()

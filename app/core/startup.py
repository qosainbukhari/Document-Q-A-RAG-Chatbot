import os

from src.components import build_components
from src.config import load_config
from src.embeddings.embedding_model import EmbeddingModel
from src.llm.groq_client import GroqClient
from src.llm.conversation_memory import SessionMemoryStore

from src.utils.logger import logger


def load_models(app, config_path=None):

    try:

        # CONFIG_PATH in .env picks the pipeline (e.g. a naive one) without code changes
        config_path = config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")

        logger.info(f"Using config: {config_path}")

        config = load_config(config_path)

        logger.info("Loading embedding model...")

        embedding_model = EmbeddingModel().get_model()

        logger.info("Loading LLM...")

        llm = GroqClient().get_llm()

        logger.info("Building RAG pipeline...")

        components = build_components(config, embedding_model, llm)

        logger.info("Loading conversation memory...")

        memory_store = SessionMemoryStore(
            window_size=config["memory"]["window_size"],
            max_sessions=config["memory"].get("max_sessions", 1000)
        )

        # Store globally inside FastAPI app
        app.state.components = components
        app.state.memory_store = memory_store

        logger.info(
            "All models loaded successfully"
        )

    except Exception as e:

        logger.error(
            f"Startup loading failed: {e}"
        )

        raise

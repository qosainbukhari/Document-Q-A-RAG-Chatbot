import yaml

from src.embeddings.embedding_model import EmbeddingModel
from src.vectordb.chroma_store import ChromaVectorStore
from src.llm.groq_client import GroqClient
from src.llm.conversation_memory import SessionMemoryStore

from src.utils.logger import logger


def load_models(app, config_path="configs/config.yaml"):

    try:

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        logger.info("Loading embedding model...")

        embedding_model = EmbeddingModel().get_model()

        logger.info("Loading vector database...")

        vector_store = ChromaVectorStore(embedding_model)

        vectordb = vector_store.load_vector_store()

        logger.info("Loading LLM...")

        llm = GroqClient().get_llm()

        logger.info("Loading conversation memory...")

        memory_store = SessionMemoryStore(
            window_size=config["memory"]["window_size"],
            max_sessions=config["memory"].get("max_sessions", 1000)
        )

        # Store globally inside FastAPI app
        app.state.embedding_model = embedding_model
        app.state.vector_store = vector_store
        app.state.vectordb = vectordb
        app.state.llm = llm
        app.state.memory_store = memory_store

        logger.info(
            "All models loaded successfully"
        )

    except Exception as e:

        logger.error(
            f"Startup loading failed: {e}"
        )

        raise

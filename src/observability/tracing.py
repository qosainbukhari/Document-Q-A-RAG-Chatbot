import os
import time
from contextlib import ExitStack, contextmanager

from src.utils.logger import logger


class Span:

    # What a traced step gets: call .update(output=..., metadata=...).
    # Without Langfuse it simply does nothing.

    def __init__(self, observation=None):

        self.observation = observation

    def update(self, **fields):

        if self.observation is None:
            return

        try:
            self.observation.update(**fields)
        except Exception as e:
            logger.warning(f"Langfuse update failed: {e}")


class Tracer:

    # Optional Langfuse tracing. Every step of a question becomes a
    # nested "span" (rewrite -> retrieval -> rerank -> generation) with
    # its input, output and latency, viewable in the Langfuse UI.
    # No LANGFUSE keys in .env -> tracing is off and the app runs normally.

    def __init__(self):

        self.client = None

        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            logger.info("Langfuse keys not set: tracing disabled")
            return

        try:
            from langfuse import get_client, propagate_attributes

            self.client = get_client()
            self.propagate_attributes = propagate_attributes

            logger.info("Langfuse tracing enabled")

        except Exception as e:
            logger.warning(f"Langfuse unavailable, tracing disabled: {e}")

    @property
    def enabled(self):

        return self.client is not None

    @contextmanager
    def span(self, name, as_type="span", input=None, session_id=None):

        start = time.perf_counter()

        with ExitStack() as stack:

            observation = None

            if self.client:
                try:
                    observation = stack.enter_context(
                        self.client.start_as_current_observation(
                            name=name, as_type=as_type, input=input
                        )
                    )

                    # Groups all questions of one chat session together
                    if session_id:
                        stack.enter_context(
                            self.propagate_attributes(session_id=session_id)
                        )

                except Exception as e:
                    logger.warning(f"Langfuse span failed: {e}")

            span = Span(observation)

            try:
                yield span

            finally:
                latency_ms = round((time.perf_counter() - start) * 1000)

                span.update(metadata={"latency_ms": latency_ms})

                logger.info(f"[trace] {name}: {latency_ms} ms")

    def flush(self):

        # Langfuse sends spans in the background; push the rest on shutdown
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning(f"Langfuse flush failed: {e}")


_tracer = None


def get_tracer():

    # One shared tracer, created on first use (after .env is loaded)
    global _tracer

    if _tracer is None:
        _tracer = Tracer()

    return _tracer

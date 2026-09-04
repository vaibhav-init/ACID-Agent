"""ACID-compliant data agent — implementation of the Agentic Transactions paper."""

from .tracing import configure as _configure_tracing

__version__ = "0.1.0"

# Push LANGSMITH_* from .env into os.environ before langchain-core reads it, so
# every entry point (CLI, scripts, pytest) traces consistently.
TRACING_ENABLED = _configure_tracing()
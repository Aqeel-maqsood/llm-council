"""Configuration for the LLM Council."""
import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic API key
ANTHROPIC_API_KEY = os.getenv("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")

# Council members - Anthropic model identifiers
COUNCIL_MODELS = [
    "claude-haiku-4-5-20251001",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "claude-haiku-4-5-20251001"

# Anthropic API endpoint
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

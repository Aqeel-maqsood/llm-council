"""Anthropic API client for making LLM requests."""
import httpx
from typing import List, Dict, Any, Optional
from .config import ANTHROPIC_API_KEY, ANTHROPIC_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via the Anthropic API.

    Args:
        model: Anthropic model identifier (e.g., "claude-haiku-4-5-20251001")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # Anthropic takes any "system" message as a top-level field, not inside messages
    system_prompt = None
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            chat_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": chat_messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            # Anthropic returns content as a list of blocks; join the text ones
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            return {
                'content': text,
                'reasoning_details': None
            }
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of Anthropic model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio
    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]
    # Wait for all to complete
    responses = await asyncio.gather(*tasks)
    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}

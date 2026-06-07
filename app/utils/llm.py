import httpx
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

async def query_llm(
    system_instruction: str,
    user_prompt: str,
    json_mode: bool = False,
    model_override: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    disable_thinking: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Unified LLM call orchestrator.
    Queries the OpenRouter API with the primary model or override model.
    Pass disable_thinking=True for fast data-filtering tasks that don't need reasoning.
    Supports tool calling if tools are provided.
    """
    api_key = settings.OPENROUTER_API_KEY_ELIGIBILITY or settings.OPENROUTER_API_KEY_NETWORKING
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lighthouse.opportunityos.github.io",
        "X-Title": "OpportunityOS"
    }

    model_name = model_override or settings.MODEL_PRIMARY
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if history:
        # Pass through history directly
        messages.extend(history)
    else:
        messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "reasoning": {"enabled": not disable_thinking}
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    try:
        logger.info(f"Querying OpenRouter API with model={model_name}, json_mode={json_mode}, reasoning={not disable_thinking}, tools={bool(tools)}")
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(settings.OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code == 200:
                res_json = response.json()
                logger.info(f"OpenRouter response received successfully.")
                choice = res_json.get("choices", [{}])[0]
                message = choice.get("message", {})

                # Check for tool calls
                if message.get("tool_calls"):
                    # For now, we return the whole message dict if there are tool calls
                    # so the caller can handle them. 
                    # Alternatively, we could handle them here, but endpoints.py might want to manage the loop.
                    # To maintain backward compatibility with (str, dict) return type, 
                    # we'll return a special string or handle it in endpoints.py.
                    return json.dumps(message), None

                content = message.get("content")
                if content is None:
                    content = ""
                content = content.strip()

                reasoning_details = message.get("reasoning_details")
                reasoning_trace = {"trace": reasoning_details} if reasoning_details else None
                return content, reasoning_trace
            else:
                raise Exception(f"OpenRouter returned {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"OpenRouter query failed: {str(e)}")
        raise e

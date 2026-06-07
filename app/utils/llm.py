import httpx
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global index for key rotation
_cerebras_key_index = 0

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
    Switches between providers (Cerebras, OpenRouter) based on LLM_PROVIDER setting.
    Rotates through Cerebras API keys with automatic fallback on rate limits.
    """
    global _cerebras_key_index
    
    provider = settings.LLM_PROVIDER
    
    if provider == "cerebras":
        cerebras_keys = settings.CEREBRAS_API_KEYS
        if not cerebras_keys:
            logger.warning("Cerebras provider selected but no keys found. Falling back to OpenRouter.")
            provider = "openrouter"
        else:
            # Try all available Cerebras keys before giving up
            num_keys = len(cerebras_keys)
            for attempt in range(num_keys):
                current_index = _cerebras_key_index % num_keys
                api_key = cerebras_keys[current_index]
                # Always increment for the next call
                _cerebras_key_index += 1
                
                model_name = model_override or settings.CEREBRAS_MODEL
                url = settings.CEREBRAS_URL
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                if history:
                    messages.extend(history)
                else:
                    messages.append({"role": "user", "content": user_prompt})

                payload = {
                    "model": model_name,
                    "messages": messages,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                if tools:
                    payload["tools"] = tools
                    if tool_choice:
                        payload["tool_choice"] = tool_choice

                try:
                    logger.info(f"Querying Cerebras API (Attempt {attempt+1}/{num_keys}) using key index {current_index} with model={model_name}")
                    async with httpx.AsyncClient(timeout=45.0) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        
                        if response.status_code == 200:
                            res_json = response.json()
                            logger.info(f"Cerebras response received successfully using key index {current_index}.")
                            choice = res_json.get("choices", [{}])[0]
                            message = choice.get("message", {})

                            if message.get("tool_calls"):
                                return json.dumps(message), None

                            content = message.get("content", "").strip()
                            reasoning_details = message.get("reasoning_details")
                            reasoning_trace = {"trace": reasoning_details} if reasoning_details else None
                            return content, reasoning_trace
                        
                        elif response.status_code == 429:
                            logger.warning(f"Cerebras Key {current_index} rate limited (429). Trying next key...")
                            continue
                        else:
                            logger.error(f"Cerebras API error (Key {current_index}): {response.status_code} - {response.text}")
                            # For non-rate-limit errors, we might still want to try the next key
                            continue
                            
                except Exception as e:
                    logger.error(f"Cerebras request failed (Key {current_index}): {str(e)}")
                    continue
            
            # If we reach here, all Cerebras keys failed
            logger.error("All Cerebras API keys failed or were rate limited. Falling back to OpenRouter.")
            provider = "openrouter"

    # OpenRouter Logic (Fallback or Primary)
    if provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY_ELIGIBILITY or settings.OPENROUTER_API_KEY_NETWORKING
        url = settings.OPENROUTER_URL
        model_name = model_override or settings.MODEL_PRIMARY
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lighthouse.opportunityos.github.io",
            "X-Title": "OpportunityOS"
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if history:
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
            logger.info(f"Querying OpenRouter API with model={model_name}, json_mode={json_mode}")
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    res_json = response.json()
                    logger.info("OpenRouter response received successfully.")
                    choice = res_json.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    if message.get("tool_calls"):
                        return json.dumps(message), None

                    content = message.get("content", "").strip()
                    reasoning_details = message.get("reasoning_details")
                    reasoning_trace = {"trace": reasoning_details} if reasoning_details else None
                    return content, reasoning_trace
                else:
                    raise Exception(f"OpenRouter returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"OpenRouter query failed: {str(e)}")
            raise e

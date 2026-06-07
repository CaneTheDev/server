import httpx
import logging
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

async def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a search using the Tavily API.
    Used as a tool by the AI Career Coach.
    """
    if not settings.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set. Search tool will return empty results.")
        return []

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced"
                }
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            
            # Normalise results for the LLM
            return [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    "score": r.get("score")
                }
                for r in results
            ]
    except Exception as e:
        logger.error(f"Tavily search tool failed: {str(e)}")
        return []

# Tool definition for OpenRouter/OpenAI
TAVILY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Search the web for real-time information, job postings, career advice, and industry trends.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of search results to return (default is 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

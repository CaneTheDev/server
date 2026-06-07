import os
import json
import httpx
import logging
from typing import List, Dict, Any
from app.core.config import settings
from app.schemas.analysis import DiscoveryRequest, DiscoveryResponse, DiscoveryOpportunity

logger = logging.getLogger(__name__)

async def run_discovery_agent(payload: DiscoveryRequest) -> DiscoveryResponse:
    """
    Executes the Three-Step Discovery Pipeline:
    1. Dynamic Query Generation (Prefix Strategy)
    2. Live Search (Tavily)
    3. LLM Filtering & Structuring
    """
    # --- STEP 1: Dynamic Query Generation ---
    # Append smart prefixes based on category to optimize search results
    interest_q = payload.user_interest
    if interest_q.lower() == "cs":
        interest_q = "Computer Science"
    
    location_q = f'"{payload.location}"'
    
    # Generic global sites with location constraint
    prefixes = {
        "internship": f'site:linkedin.com/jobs OR site:indeed.com OR site:glassdoor.com "internship" "{payload.academic_level}" "{interest_q}" {location_q} 2026',
        "scholarship": f'site:opportunitydesk.org OR site:scholars4dev.com "scholarship" "{payload.academic_level}" "{interest_q}" {location_q} 2026',
        "community": f'(site:t.me/ OR site:discord.gg/ OR site:slack.com OR site:reddit.com/r/ OR site:youtube.com) "{interest_q}" community {payload.location}',
        "networking": f'(site:eventbrite.com OR site:meetup.com OR site:linkedin.com/events OR site:youtube.com) "{interest_q}" {payload.location}',
        "fellowship": f'site:opportunitydesk.org OR site:profellow.com "fellowship" "{payload.academic_level}" "{interest_q}" {location_q} 2026'
    }

    # If location is Nigeria, we can add local job boards
    if "nigeria" in payload.location.lower():
        prefixes["internship"] = f'site:myjobmag.com OR site:hotnigerianjobs.com OR site:linkedin.com/jobs "internship" "{payload.academic_level}" "{interest_q}" {location_q} 2026'
    
    search_query = prefixes.get(payload.category.lower(), f'"{interest_q}" "{payload.location}" 2026')
    logger.info(f"Discovery Agent: Generated search query: {search_query}")
    
    # --- STEP 2: Live Search Execution ---
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Call Tavily Search API
            tavily_response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": search_query,
                    "search_depth": "basic",
                    "max_results": 20 # Increased to 20 to ensure we have enough pool for 10 filtered results
                }
            )
            tavily_response.raise_for_status()
            raw_results = tavily_response.json().get("results", [])
            logger.info(f"Discovery Agent: Found {len(raw_results)} raw results from Tavily.")
            if raw_results:
                logger.info(f"Discovery Agent: Sample result title: {raw_results[0].get('title')}")
            
            if not raw_results:
                logger.warning("Discovery Agent: No raw results found from Tavily.")
                return DiscoveryResponse(opportunities=[])
                
            # --- STEP 3: LLM Filtering & Structuring ---
            exclude_prompt = ""
            if payload.exclude_urls:
                exclude_prompt = f"\nCRITICAL: Do NOT include any of the following URLs as they have already been shown to the user:\n{json.dumps(payload.exclude_urls)}"

            system_prompt = (
                "You are a highly efficient data extraction agent. Your goal is to convert search results into a clean JSON list of opportunities.\n\n"
                f"TARGET: {payload.user_interest} {payload.category} in {payload.location}.\n\n"
                "INSTRUCTIONS:\n"
                "1. Analyze the search results and extract up to 10 relevant items.\n"
                "2. For each item, you MUST provide these 5 keys: 'title', 'url', 'organization', 'description', 'availability'.\n"
                "3. Use the following mapping:\n"
                "   - title: The name of the position, event, or scholarship.\n"
                "   - url: The direct link from the search result.\n"
                "   - organization: The host company, university, or platform name.\n"
                "   - description: A concise summary of the opportunity (max 2 sentences).\n"
                "   - availability: Set to 'Active 2026' or 'Ongoing'.\n"
                "4. CRITICAL: Do NOT return empty keys or null values. If you find results, map them correctly.\n"
                f"{exclude_prompt}\n"
                "JSON FORMAT (Strict):\n"
                "{\n"
                '  "opportunities": [\n'
                '    {\n'
                '      "title": "...",\n'
                '      "url": "...",\n'
                '      "organization": "...",\n'
                '      "description": "...",\n'
                '      "availability": "..."\n'
                '    }\n'
                '  ]\n'
                "}"
            )
            
            user_prompt = f"Raw Search Results:\n{json.dumps(raw_results)}\n\nExtract all relevant opportunities into JSON."
            
            logger.info(f"Discovery Agent: Sending {len(raw_results)} results to LLM for filtering.")
            
            from app.utils.llm import query_llm
            content, reasoning_trace = await query_llm(
                system_instruction=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                disable_thinking=False # Enable reasoning for better extraction
            )
            
            logger.info(f"Discovery Agent: LLM returned content (first 200 chars): {content[:200]}...")
            
            # Clean up potential markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).replace("```", "", 1).strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 1).replace("```", "", 1).strip()
                
            structured_data = json.loads(content)
            
            raw_opp_data = structured_data.get("opportunities", [])
            opp_data = []
            
            # --- VALIDATION: Filter out malformed items ---
            required_keys = {"title", "url", "organization", "description", "availability"}
            for opp in raw_opp_data:
                if isinstance(opp, dict) and all(k in opp and opp[k] for k in required_keys):
                    opp_data.append(opp)
                else:
                    logger.warning(f"Discovery Agent: Skipping malformed opportunity object: {opp}")
            
            # --- FALLBACK: If LLM returns nothing or trash, manually pick top 5 results ---
            if not opp_data and raw_results:
                logger.warning("Discovery Agent: LLM returned no valid opportunities. Falling back to raw results.")
                for res in raw_results[:8]: # Increase to 8 for fallback
                    title = res.get("title", "Found Opportunity")
                    url = res.get("url", "")
                    if not url: continue
                    
                    opp_data.append({
                        "title": title,
                        "url": url,
                        "organization": "Search Result",
                        "description": res.get("content", "Details available at link.")[:300],
                        "availability": "Active"
                    })
            
            opportunities = [
                DiscoveryOpportunity(**opp) for opp in opp_data[:10] # Cap at 10
            ]
            
            logger.info(f"Discovery Agent: Successfully parsed {len(opportunities)} opportunities.")
            
            return DiscoveryResponse(opportunities=opportunities)
            
        except Exception as e:
            logger.error(f"Discovery Agent: Live search or filtering failed: {str(e)}")
            # Return empty response instead of failing the whole request
            return DiscoveryResponse(opportunities=[])

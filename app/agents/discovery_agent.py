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
    Executes the Enhanced Discovery Pipeline:
    1. CV Information Extraction (if provided)
    2. Dynamic Open Search Query Generation
    3. Live Search (Tavily)
    4. LLM Extraction, Filtering & Insight Generation
    """
    from app.utils.llm import query_llm
    
    cv_extraction_summary = ""
    effective_interest = payload.user_interest
    effective_major = payload.major
    effective_level = payload.academic_level
    effective_skills = payload.skills or []

    # --- STEP 1: CV Extraction (Optional) ---
    if payload.cv_text:
        logger.info("Discovery Agent: Extracting information from CV text.")
        cv_system_prompt = (
            "You are a professional CV analyzer. Extract the following details from the provided CV text:\n"
            "1. Core Interests/Target Role (one short phrase)\n"
            "2. Academic Major/Field\n"
            "3. Academic Level (e.g., University, Graduate, PhD, Professional)\n"
            "4. Top 5 Skills (list)\n\n"
            "Return a JSON object with keys: 'interest', 'major', 'level', 'skills'."
        )
        cv_user_prompt = f"CV Text:\n{payload.cv_text[:4000]}" # Limit text size
        
        try:
            cv_content, _ = await query_llm(
                system_instruction=cv_system_prompt,
                user_prompt=cv_user_prompt,
                json_mode=True
            )
            cv_data = json.loads(cv_content)
            effective_interest = cv_data.get('interest', effective_interest)
            effective_major = cv_data.get('major', effective_major)
            effective_level = cv_data.get('level', effective_level)
            effective_skills = cv_data.get('skills', effective_skills)
            
            cv_extraction_summary = (
                f"Extracted from CV: {effective_interest} focus, {effective_major} background, "
                f"Level: {effective_level}, Skills: {', '.join(effective_skills[:3])}."
            )
            logger.info(f"Discovery Agent: CV Extraction successful. {cv_extraction_summary}")
        except Exception as e:
            logger.error(f"Discovery Agent: CV Extraction failed: {str(e)}")

    # --- STEP 2: Dynamic Open Search Query Generation ---
    interest_q = effective_interest
    if interest_q.lower() == "cs":
        interest_q = "Computer Science"
    
    location_q = payload.location
    if not location_q or location_q.lower() == "remote":
        location_q = "Global Remote"
    
    # Open search strategy: No site: restrictions to allow for niche/remote discovery
    # Use broad keywords to capture diverse opportunities
    search_query = f'"{interest_q}" {payload.category} OR jobs OR opportunities in "{location_q}" 2026'
    
    # If location is specific, add it as a required term
    if payload.location and payload.location.lower() != "remote":
        search_query = f'"{interest_q}" {payload.category} OR "career opportunities" "{payload.location}" 2026'

    logger.info(f"Discovery Agent: Generated open search query: {search_query}")
    
    # --- STEP 3: Live Search Execution ---
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            tavily_response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": search_query,
                    "search_depth": "advanced", # Use advanced for deeper open search
                    "max_results": 20
                }
            )
            tavily_response.raise_for_status()
            raw_results = tavily_response.json().get("results", [])
            logger.info(f"Discovery Agent: Found {len(raw_results)} raw results.")
            
            if not raw_results:
                logger.warning("Discovery Agent: No raw results found.")
                return DiscoveryResponse(opportunities=[], ai_comment="I couldn't find any live results for this specific query. Try broadening your location or role interest.")
                
            # --- STEP 4: LLM Extraction & Filtering ---
            exclude_prompt = ""
            if payload.exclude_urls:
                exclude_prompt = f"\nCRITICAL: Do NOT include any of the following URLs as they have already been shown to the user:\n{json.dumps(payload.exclude_urls)}"

            extraction_system_prompt = (
                "You are a highly efficient data extraction and career matching agent. Your goal is to convert search results into a clean JSON list of opportunities.\n\n"
                f"TARGET: {effective_interest} {payload.category} in {payload.location}.\n"
                f"USER PROFILE: Major: {effective_major}, Academic Level: {effective_level}, Skills: {', '.join(effective_skills)}.\n\n"
                "INSTRUCTIONS:\n"
                "1. Analyze the search results and extract up to 10 relevant items.\n"
                "2. TAILORING: ONLY include results that are a good fit for the user's major and academic level. If a result is irrelevant, skip it.\n"
                "3. DE-DUPLICATION: Strictly avoid duplicate opportunities. If multiple results point to the same role or organization, pick the best one.\n"
                "4. For each item, provide: 'title', 'url', 'organization', 'description', 'availability'.\n"
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
            
            extraction_content, _ = await query_llm(
                system_instruction=extraction_system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                disable_thinking=False
            )
            
            if extraction_content.startswith("```json"):
                extraction_content = extraction_content.replace("```json", "", 1).replace("```", "", 1).strip()
            elif extraction_content.startswith("```"):
                extraction_content = extraction_content.replace("```", "", 1).replace("```", "", 1).strip()
                
            structured_data = json.loads(extraction_content)
            raw_opp_data = structured_data.get("opportunities", [])
            
            required_keys = {"title", "url", "organization", "description", "availability"}
            opp_data = []
            for opp in raw_opp_data:
                if isinstance(opp, dict) and all(k in opp and opp[k] for k in required_keys):
                    opp_data.append(opp)
            
            if not opp_data and raw_results:
                for res in raw_results[:8]:
                    title = res.get("title", "Found Opportunity")
                    url = res.get("url", "")
                    if not url: continue
                    opp_data.append({
                        "title": title, "url": url, "organization": "Search Result",
                        "description": res.get("content", "Details available at link.")[:300],
                        "availability": "Active"
                    })

            # --- STEP 5: LLM Insight Generation (Post-Extraction) ---
            insight_system_prompt = (
                "You are a career advisor providing context for a list of job/scholarship opportunities.\n"
                "Your goal is to provide a concise (1-2 sentences) 'ai_comment' based on the results being shown to the user.\n\n"
                f"CONTEXT:\n"
                f"- Target: {effective_interest} {payload.category} in {payload.location}\n"
                f"- User Profile: Major: {effective_major}, Academic Level: {effective_level}\n"
                f"- {cv_extraction_summary}\n\n"
                "INSTRUCTIONS:\n"
                "1. If a CV was used, acknowledge what was extracted and how it influenced the search.\n"
                "2. Look at the 'Final Results' being sent to the user.\n"
                "3. If results are scarce, explain why (niche area, specific location).\n"
                "4. Provide an encouraging insight about the roles found.\n"
                "5. Return a JSON object with a single key 'ai_comment'."
            )
            
            insight_user_prompt = (
                f"Final Results being shown: {json.dumps(opp_data)}\n\n"
                "Generate the ai_comment JSON."
            )
            
            insight_content, _ = await query_llm(
                system_instruction=insight_system_prompt,
                user_prompt=insight_user_prompt,
                json_mode=True,
                disable_thinking=False
            )

            if insight_content.startswith("```json"):
                insight_content = insight_content.replace("```json", "", 1).replace("```", "", 1).strip()
            elif insight_content.startswith("```"):
                insight_content = insight_content.replace("```", "", 1).replace("```", "", 1).strip()

            try:
                insight_data = json.loads(insight_content)
                ai_comment = insight_data.get("ai_comment")
            except:
                ai_comment = f"{cv_extraction_summary} Found some matches for you."

            opportunities = [DiscoveryOpportunity(**opp) for opp in opp_data[:10]]
            return DiscoveryResponse(opportunities=opportunities, ai_comment=ai_comment)
            
        except Exception as e:
            logger.error(f"Discovery Agent: Live search or filtering failed: {str(e)}")
            # Return empty response instead of failing the whole request
            return DiscoveryResponse(opportunities=[])

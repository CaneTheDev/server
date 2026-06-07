import logging
import asyncio
import json
from fastapi import APIRouter, HTTPException
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, ChatRequest, ChatResponse, DiscoveryRequest, DiscoveryResponse
from app.agents.eligibility import evaluate_eligibility
from app.agents.networking import evaluate_networking
from app.agents.discovery_agent import run_discovery_agent
from app.utils.tools import TAVILY_SEARCH_TOOL, tavily_search

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/discover/live", response_model=DiscoveryResponse)
async def discover_live(payload: DiscoveryRequest):
    """
    Triggers the live discovery agent to fetch and filter opportunities
    from the web based on category and user interests.
    """
    try:
        logger.info(f"Received discovery request: Category={payload.category}, Interest={payload.user_interest}, Location={payload.location}")
        result = await run_discovery_agent(payload)
        return result
    except Exception as e:
        logger.exception("Error during live discovery")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_opportunity(payload: AnalysisRequest):
    """
    Stateless endpoint that evaluates an opportunity for a given user profile.
    Orchestrates Branch A (Eligibility) and Branch B (Networking) concurrently.
    """
    try:
        logger.info(f"Received analysis request for User={payload.profile.name}, Opportunity={payload.opportunity.title}")
        
        # Run Branch A and Branch B in parallel
        eligibility_task = evaluate_eligibility(payload.profile, payload.opportunity)
        networking_task = evaluate_networking(payload.profile, payload.opportunity)
        
        eligibility_res, networking_res = await asyncio.gather(eligibility_task, networking_task)
        
        parsed_eligibility, eligibility_reasoning = eligibility_res
        contacts, networking_reasoning = networking_res
        
        # Combine reasoning details
        reasoning_details = {
            "eligibility": eligibility_reasoning.get("trace") if eligibility_reasoning else None,
            "networking": networking_reasoning if networking_reasoning else None
        }

        response_data = AnalysisResponse(
            opp_id=payload.opportunity.id,
            match_score=parsed_eligibility.get("match_score", 50),
            success_probability=parsed_eligibility.get("success_probability", "Medium"),
            eligibility_reasoning=parsed_eligibility.get("eligibility_reasoning", ""),
            application_tips=parsed_eligibility.get("application_tips", []),
            suggested_contacts=contacts,
            reasoning_details=reasoning_details
        )
        
        logger.info("Analysis completed successfully.")
        return response_data

    except Exception as e:
        logger.exception("Error processing opportunity analysis")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_coach(payload: ChatRequest):
    """
    Processes follow-up conversation with the AI Career Coach.
    Accepts conversation history to support stateful reasoning.
    """
    from app.utils.llm import query_llm

    try:
        logger.info("Received coaching chat request")
        
        # Construct system prompt context
        system_content = (
            "You are the OpportunityOS AI Career Coach and Academic Advisor. "
            "Help the candidate prepare for their opportunity, answer follow-up questions about their match score, "
            "suggest resources, and refine networking strategies. "
            "Be extremely friendly, professional, practical, and detail-oriented. Keep responses concise.\n\n"
            "JOB SEARCH PROTOCOL:\n"
            "1. If the user mentions they are looking for a job or career opportunity, "
            "IMMEDIATELY ask them to share their CV or highlight their key skills/experiences if they haven't already.\n"
            "2. DO NOT just suggest general job boards. Use the 'tavily_search' tool to find REAL, CURRENT job openings "
            "that match their profile and interests.\n"
            "3. When presenting jobs found via search, provide the title, company, and a direct link.\n"
            "4. You have access to real-time web search. Always prefer finding fresh data over giving generic advice."
        )
        
        if payload.profile:
            system_content += (
                f"\n\nCandidate Profile:\n"
                f"- Name: {payload.profile.name}\n"
                f"- Major: {payload.profile.major}\n"
                f"- GPA: {payload.profile.gpa}\n"
                f"- Skills: {', '.join(payload.profile.skills)}\n"
                f"- Interests: {', '.join(payload.profile.interests)}\n"
                f"- Location: {payload.profile.location}"
            )
        if payload.opportunity:
            system_content += (
                f"\n\nTarget Opportunity:\n"
                f"- Title: {payload.opportunity.title}\n"
                f"- Organization: {payload.opportunity.organization}\n"
                f"- Requirements: {payload.opportunity.requirements}"
            )

        # Build history for query_llm (skip system messages)
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in payload.messages
            if msg.role != "system"
        ]

        # Define tools
        tools = [TAVILY_SEARCH_TOOL]

        # The last user message is the prompt; rest is history
        # query_llm accepts history and uses the last entry as user_prompt internally
        content, reasoning_trace = await query_llm(
            system_instruction=system_content,
            user_prompt="",  # Not used when history is provided
            json_mode=False,
            history=history,
            tools=tools
        )

        # Handle potential tool calls
        try:
            # If content is a JSON string, it might be a message with tool_calls
            if content.startswith('{') and '"tool_calls"' in content:
                message_dict = json.loads(content)
                if message_dict.get("tool_calls"):
                    tool_calls = message_dict["tool_calls"]
                    # Add assistant's tool call message to history
                    history.append(message_dict)
                    
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        if function_name == "tavily_search":
                            logger.info(f"Executing tool: tavily_search with args: {arguments}")
                            search_results = await tavily_search(**arguments)
                            # Add tool response to history
                            history.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": function_name,
                                "content": json.dumps(search_results)
                            })
                    
                    # Second call with tool results
                    content, reasoning_trace = await query_llm(
                        system_instruction=system_content,
                        user_prompt="",
                        json_mode=False,
                        history=history,
                        tools=tools
                    )
        except Exception as tool_err:
            logger.error(f"Error during tool execution loop: {str(tool_err)}")
            # Fallback: if tool execution fails, we might already have 'content' from the first call 
            # or we just let it be.

        return ChatResponse(
            role="assistant",
            content=content,
            reasoning_details=reasoning_trace
        )

    except Exception as e:
        logger.exception("Error processing coach chat")
        # Graceful fallback instead of 500 error
        return ChatResponse(
            role="assistant",
            content="I'm here to support you! It looks like our AI connection is a bit slow right now. How else can I help you improve your application profile?",
            reasoning_details={"trace": "Coaching API fallback activated."}
        )

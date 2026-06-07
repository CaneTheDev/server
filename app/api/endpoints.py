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

def parse_llm_response(content_str: str):
    """
    Parses content_str which could be a raw text response or a JSON string (well-formed or malformed).
    Returns (tool_calls, text_content, reasoning_trace).
    """
    from typing import List, Dict, Any, Optional, Tuple
    import re
    import json

    cleaned = content_str.strip()
    
    # Strip markdown code blocks if the model wrapped the JSON
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # If it doesn't look like JSON, treat it as raw text content
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        return None, content_str, None

    # Try standard JSON parsing first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            tool_calls = data.get("tool_calls")
            text_content = data.get("content") or ""
            reasoning = data.get("reasoning")
            reasoning_trace = {"trace": reasoning} if reasoning else None
            
            # If we got tool_calls or text content or reasoning, return them
            if tool_calls or text_content or reasoning:
                return tool_calls, text_content, reasoning_trace
    except json.JSONDecodeError:
        pass

    # Fallback/robust parsing for malformed JSON (e.g. unescaped quotes)
    tool_calls = None
    tc_idx = cleaned.find('"tool_calls"')
    if tc_idx != -1:
        start_arr = cleaned.find('[', tc_idx)
        if start_arr != -1:
            # Find matching ']'
            bracket_count = 1
            end_arr = -1
            for i in range(start_arr + 1, len(cleaned)):
                if cleaned[i] == '[':
                    bracket_count += 1
                elif cleaned[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_arr = i
                        break
            if end_arr != -1:
                array_str = cleaned[start_arr + 1 : end_arr].strip()
                tool_calls = []
                idx = 0
                while idx < len(array_str):
                    start_obj = array_str.find('{', idx)
                    if start_obj == -1:
                        break
                    brace_count = 1
                    end_obj = -1
                    for j in range(start_obj + 1, len(array_str)):
                        if array_str[j] == '{':
                            brace_count += 1
                        elif array_str[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_obj = j
                                break
                    if end_obj == -1:
                        break
                    obj_str = array_str[start_obj : end_obj + 1]
                    
                    id_match = re.search(r'"id"\s*:\s*"([^"]+)"', obj_str)
                    call_id = id_match.group(1) if id_match else "unknown_id"
                    
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', obj_str)
                    fn_name = name_match.group(1) if name_match else "unknown_function"
                    
                    args_dict = {}
                    args_key_idx = obj_str.find('"arguments"')
                    if args_key_idx != -1:
                        start_args = obj_str.find('{', args_key_idx)
                        if start_args != -1:
                            brace_count = 1
                            end_args = -1
                            for k in range(start_args + 1, len(obj_str)):
                                if obj_str[k] == '{':
                                    brace_count += 1
                                elif obj_str[k] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_args = k
                                        break
                            if end_args != -1:
                                args_str = obj_str[start_args : end_args + 1]
                                try:
                                    args_dict = json.loads(args_str)
                                except Exception:
                                    pairs = re.findall(r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+)', args_str)
                                    for pk, pv in pairs:
                                        if pv.startswith('"') and pv.endswith('"'):
                                            args_dict[pk] = pv[1:-1]
                                        else:
                                            try:
                                                args_dict[pk] = int(pv)
                                            except ValueError:
                                                args_dict[pk] = pv
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(args_dict)
                        }
                    })
                    idx = end_obj + 1

    # 2. Extract reasoning (robust against unescaped double quotes inside value)
    reasoning = None
    reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*(?:,\s*"(?:tool_calls|role|content)"|\})', cleaned, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1)
    else:
        # Fallback to simple matching if keys are ordered differently or missing
        reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*(?:,|\})', cleaned, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
    
    # 3. Extract content (robust against unescaped double quotes inside value)
    text_content = ""
    content_match = re.search(r'"content"\s*:\s*"(.*?)"\s*(?:,\s*"(?:tool_calls|role|reasoning)"|\})', cleaned, re.DOTALL)
    if content_match:
        text_content = content_match.group(1)
    else:
        # Fallback to simple matching
        content_match = re.search(r'"content"\s*:\s*"(.*?)"\s*(?:,|\})', cleaned, re.DOTALL)
        if content_match:
            text_content = content_match.group(1)

    if text_content:
        # Clean up escapes
        text_content = text_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
    elif not tool_calls:
        # No tool calls, and could not extract content field, return original string
        return None, content_str, None

    reasoning_trace = {"trace": reasoning} if reasoning else None
    return tool_calls, text_content, reasoning_trace

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
            "4. You have access to real-time web search. Always prefer finding fresh data over giving generic advice.\n\n"
            "SEARCH EFFICIENCY PROTOCOL:\n"
            "1. Minimize sequential search calls. Try to structure a single, high-quality search query first.\n"
            "2. You are limited to a maximum of 2 search calls per user turn. Do not make multiple searches with minor variations of the same query.\n"
            "3. If your search does not yield perfect matches, do not keep searching. Instead, present the best available findings and offer advice on how the candidate can search further or ask them for more details."
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

        # Handle potential tool calls in a loop (up to max_loops to handle sequential tool steps)
        try:
            loop_count = 0
            max_loops = 5
            while loop_count < max_loops:
                tool_calls, text_content, reasoning_trace = parse_llm_response(content)
                if tool_calls:
                    # Clean assistant message to only contain valid API fields
                    assistant_msg = {
                        "role": "assistant",
                        "content": text_content or None,
                        "tool_calls": tool_calls
                    }
                    history.append(assistant_msg)
                    
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        if function_name == "tavily_search":
                            logger.info(f"Executing tool: tavily_search with args: {arguments}")
                            search_results = await tavily_search(**arguments)
                            tool_content = json.dumps(search_results)
                        else:
                            logger.warning(f"Unknown tool call: {function_name}")
                            tool_content = json.dumps({"error": f"Tool '{function_name}' is not supported."})
                        
                        # Add tool response to history
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": tool_content
                        })
                    
                    # Query LLM again with tool results
                    content, reasoning_trace = await query_llm(
                        system_instruction=system_content,
                        user_prompt="",
                        json_mode=False,
                        history=history,
                        tools=tools
                    )
                    loop_count += 1
                else:
                    if text_content:
                        content = text_content
                    break
            
            # Fallback if we exceeded max_loops and still have a tool call in content
            tool_calls, text_content, reasoning_trace = parse_llm_response(content)
            if tool_calls:
                logger.warning("Exceeded max tool loops, forcing final call without tools.")
                content, reasoning_trace = await query_llm(
                    system_instruction=system_content + "\n\nPlease provide your final response directly without calling any tools.",
                    user_prompt="",
                    json_mode=False,
                    history=history,
                    tools=None
                )
                _, final_text, final_reasoning = parse_llm_response(content)
                if final_text:
                    content = final_text
                if final_reasoning:
                    reasoning_trace = final_reasoning
            else:
                if text_content:
                    content = text_content
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

import json
import logging
from typing import Dict, Any, Tuple, Optional
from app.core.config import settings
from app.schemas.analysis import UserProfile, Opportunity

logger = logging.getLogger(__name__)

async def evaluate_eligibility(
    profile: UserProfile, 
    opportunity: Opportunity
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Branch A: Structured Eligibility Evaluation using OpenRouter
    Returns:
        (parsed_json_data, reasoning_details)
    """
    system_instruction = (
        "You are an expert career counselor and academic advisor. "
        "Evaluate the user's profile against the opportunity requirements. "
        "You must return your output strictly in JSON format matching this schema:\n"
        "{\n"
        '  "match_score": integer (0 to 100),\n'
        '  "success_probability": "Low" or "Medium" or "High",\n'
        '  "eligibility_reasoning": "string explanation",\n'
        '  "application_tips": ["tip 1", "tip 2", "tip 3"]\n'
        "}"
    )
    
    user_prompt = (
        f"User Profile:\n- Major: {profile.major}\n- Academic Level: {profile.academicLevel}\n- Location: {profile.location}\n- Skills: {', '.join(profile.skills)}\n"
        f"- GPA: {profile.gpa}\n- Interests: {', '.join(profile.interests)}\n\n"
        f"Opportunity Details:\n- Title: {opportunity.title}\n- Organization: {opportunity.organization}\n"
        f"- Requirements: {opportunity.requirements}\n\n"
        f"Perform an in-depth fit assessment."
    )

    try:
        from app.utils.llm import query_llm
        content, reasoning_trace = await query_llm(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            json_mode=True
        )
        parsed_data = json.loads(content)
        logger.info("Eligibility evaluation completed successfully.")
        return parsed_data, reasoning_trace
    except Exception as e:
        logger.warning(f"Failed to query eligibility LLM: {str(e)}")

    # If all API calls fail, return a local fallback payload
    logger.error("All OpenRouter models failed for eligibility branch. Using local fallback.")
    
    # Calculate a mock score based on matches
    matched_skills = [skill for skill in profile.skills if skill.lower() in opportunity.requirements.lower()]
    match_score = 40 + min(len(matched_skills) * 15, 50)
    success_probability = "High" if match_score > 75 else ("Medium" if match_score > 50 else "Low")
    
    fallback_data = {
        "match_score": match_score,
        "success_probability": success_probability,
        "eligibility_reasoning": (
            f"Local fallback analysis: User has {len(matched_skills)} matching skills ({', '.join(matched_skills)}) "
            f"for this opportunity."
        ),
        "application_tips": [
            "Tailor your resume to emphasize skills matching the requirements: " + ", ".join(profile.skills),
            "Write a cover letter highlighting your background in " + profile.major,
            "Acquire more hands-on project experience in fields relevant to " + opportunity.organization
        ]
    }
    fallback_reasoning = {"trace": "Local fallback generator was activated because OpenRouter requests failed."}
    
    return fallback_data, fallback_reasoning

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from app.schemas.analysis import UserProfile, Opportunity, ContactResult
from app.agents.search_agent import search_professionals

logger = logging.getLogger(__name__)

async def generate_outreach_message(
    profile: UserProfile,
    opportunity: Opportunity,
    contact_name: str,
    contact_snippet: str,
    contact_source: str = "tavily"
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Calls query_llm (OpenRouter) to generate a tailored networking outreach message.
    """
    system_instruction = (
        "You are an expert career networking coach. Your task is to generate a personalized, "
        "professional outreach message that a student can send to a professional contact. "
        "The message must be warm, concise (under 100 words), direct, and reference the student's "
        "background and the contact's professional history if available. "
        "Do NOT include placeholders like '[Your Name]' or '[Major]' — replace them with the actual details. "
        "Output ONLY the raw text message, ready to be copied and sent."
    )

    # Adapt the prompt to the platform the contact was found on
    if contact_source == "github_api":
        platform_hint = "GitHub profile"
        send_platform = "GitHub or LinkedIn"
    else:
        platform_hint = "LinkedIn profile"
        send_platform = "LinkedIn"

    user_prompt = (
        f"Student Profile:\n"
        f"- Name: {profile.name}\n"
        f"- Major: {profile.major}\n"
        f"- Academic Level: {profile.academicLevel}\n"
        f"- Location: {profile.location}\n"
        f"- Skills: {', '.join(profile.skills)}\n"
        f"- Interests: {', '.join(profile.interests)}\n\n"
        f"Target Opportunity:\n"
        f"- Title: {opportunity.title}\n"
        f"- Organization: {opportunity.organization}\n\n"
        f"Professional to Connect With (found via {platform_hint}):\n"
        f"- Name: {contact_name}\n"
        f"- Profile Info: {contact_snippet}\n\n"
        f"Draft a custom, copy-pasteable outreach message that {profile.name} can send to "
        f"{contact_name} via {send_platform}."
    )

    try:
        from app.utils.llm import query_llm
        content, reasoning_trace = await query_llm(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            json_mode=False,
            disable_thinking=True
        )
        return content, reasoning_trace
    except Exception as e:
        logger.error(f"Failed to generate outreach message via LLM: {str(e)}")

    # local fallback if APIs fail
    fallback_message = (
        f"Hi {contact_name},\n\n"
        f"I hope you are well. I'm {profile.name}, a {profile.major} student from {profile.location}. "
        f"I came across your profile and was inspired by your journey to {opportunity.organization}. "
        f"I am currently preparing my application for the {opportunity.title} and would love to ask "
        f"a quick question about your experience. Thank you!"
    )
    return fallback_message, {"trace": "Local fallback outreach template activated due to API limits."}

async def evaluate_networking(
    profile: UserProfile,
    opportunity: Opportunity
) -> Tuple[List[ContactResult], Optional[Dict[str, Any]]]:
    """
    Branch B: Retrieves professionals using search agent, then generates outreach templates concurrently.
    """
    # 1. Search for people
    raw_contacts = await search_professionals(
        field=profile.major,
        organization=opportunity.organization,
        location=profile.location
    )

    contacts_results = []
    reasoning_traces = {}

    tasks = []
    for contact in raw_contacts:
        raw_title = contact.get("title", "Professional")
        # Parse name — support both " - " (LinkedIn) and " — " (GitHub API) separators
        cleaned_name = raw_title.replace(" — ", " - ").split(" - ")[0].split(" | ")[0].strip()
        contact_source = contact.get("source", "tavily")

        tasks.append(
            generate_outreach_message(
                profile=profile,
                opportunity=opportunity,
                contact_name=cleaned_name,
                contact_snippet=contact.get("content", ""),
                contact_source=contact_source
            )
        )
    
    # Run outreach generations concurrently
    results = await asyncio.gather(*tasks)

    for i, contact in enumerate(raw_contacts):
        raw_title = contact.get("title", "Professional")
        cleaned_name = raw_title.replace(" — ", " - ").split(" - ")[0].split(" | ")[0].strip()

        msg, reasoning = results[i]

        # Determine a sensible default profile URL based on source
        source = contact.get("source", "tavily")
        default_url = (
            "https://github.com" if source == "github_api"
            else "https://linkedin.com"
        )

        contacts_results.append(ContactResult(
            name=cleaned_name,
            profile_url=contact.get("url", default_url),
            snippet=contact.get("content", ""),
            suggested_message=msg
        ))

        if reasoning:
            reasoning_traces[cleaned_name] = reasoning.get("trace")

    return contacts_results, reasoning_traces

"""
search_agent.py — Dual-Source Professional Search
Runs queries across:
  1. Tavily LinkedIn  — `site:linkedin.com/in/ "{organization}" "{field}" "{location}"`
  2. Tavily GitHub    — `site:github.com/ "{organization}" "location:{location}"`
  3. GitHub REST API  — searches public users by org + location (uses GITHUB_TOKEN if available)

Results from all three sources are merged and deduplicated before being passed to
the Networking Agent (Branch B) for outreach drafting.
"""
import logging
import asyncio
import httpx
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TAVILY QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _tavily_query(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 2
) -> List[Dict[str, Any]]:
    """Fire a single Tavily search and return normalised result dicts."""
    try:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
            },
            timeout=12.0,
        )
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "title": r.get("title", "Profile"),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "source": "tavily",
                }
                for r in data.get("results", [])
            ]
        logger.warning(
            f"Tavily non-200 for query '{query[:60]}…': {response.status_code} — {response.text[:200]}"
        )
    except Exception as exc:
        logger.error(f"Tavily request failed for query '{query[:60]}…': {exc}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB REST API HELPER
# ─────────────────────────────────────────────────────────────────────────────

async def _github_user_search(
    client: httpx.AsyncClient,
    organization: str,
    location: str,
    max_results: int = 2
) -> List[Dict[str, Any]]:
    """
    Searches GitHub's public user API for people associated with the target
    organisation and location.  Falls back gracefully if the token is absent.
    """
    if not settings.GITHUB_TOKEN:
        logger.info("GITHUB_TOKEN not set — skipping GitHub REST user search.")
        return []

    # GitHub search query: users who have 'organization' in their bio/company
    # and are in the target location
    search_query = f"type:user location:{location} {organization} in:bio"
    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params = {
        "q": search_query,
        "per_page": max_results,
        "sort": "followers",
        "order": "desc",
    }

    try:
        response = await client.get(
            settings.GITHUB_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10.0,
        )
        if response.status_code == 200:
            items = response.json().get("items", [])
            results = []
            for user in items:
                login = user.get("login", "")
                profile_url = user.get("html_url", f"https://github.com/{login}")
                # Fetch the user's full profile for richer bio text
                bio = ""
                name = login
                try:
                    profile_resp = await client.get(
                        f"https://api.github.com/users/{login}",
                        headers=headers,
                        timeout=8.0,
                    )
                    if profile_resp.status_code == 200:
                        pdata = profile_resp.json()
                        name = pdata.get("name") or login
                        company = pdata.get("company", "")
                        bio = pdata.get("bio") or ""
                        if company:
                            bio = f"{company}. {bio}".strip()
                except Exception:
                    pass

                results.append({
                    "title": f"{name} — GitHub ({login})",
                    "url": profile_url,
                    "content": bio or f"Active GitHub contributor. Profile: {profile_url}",
                    "source": "github_api",
                })
            logger.info(f"GitHub API returned {len(results)} user(s) for '{organization}' in '{location}'.")
            return results
        else:
            logger.warning(
                f"GitHub user search returned {response.status_code}: {response.text[:200]}"
            )
    except Exception as exc:
        logger.error(f"GitHub REST API search failed: {exc}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def search_professionals(
    field: str,
    organization: str,
    location: str,
) -> List[Dict[str, Any]]:
    """
    Runs three concurrent searches:
      • Tavily — LinkedIn profiles matching organisation + field + location
      • Tavily — GitHub profiles matching organisation + location
      • GitHub REST API — public users matching organisation + location

    Merges and deduplicated by URL.  Falls back to curated mock profiles if
    all live sources return empty.
    """
    if settings.TAVILY_API_KEY:
        linkedin_query = (
            f'site:linkedin.com/in/ "{organization}" "{field}" "{location}"'
        )
        github_tavily_query = (
            f'site:github.com/ "{organization}" "location:{location}"'
        )

        async with httpx.AsyncClient() as client:
            linkedin_task = _tavily_query(client, linkedin_query, max_results=2)
            gh_tavily_task = _tavily_query(client, github_tavily_query, max_results=2)
            gh_api_task = _github_user_search(client, organization, location, max_results=2)

            linkedin_results, gh_tavily_results, gh_api_results = await asyncio.gather(
                linkedin_task, gh_tavily_task, gh_api_task
            )

        # Merge sources; prefer LinkedIn first, then GitHub REST, then Tavily-GitHub
        all_results = linkedin_results + gh_api_results + gh_tavily_results

        # Deduplicate by URL (keep first occurrence)
        seen_urls: set = set()
        deduplicated = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(r)

        if deduplicated:
            logger.info(
                f"Search returned {len(deduplicated)} unique profile(s) across all sources "
                f"(LinkedIn Tavily={len(linkedin_results)}, GitHub API={len(gh_api_results)}, "
                f"GitHub Tavily={len(gh_tavily_results)})."
            )
            return deduplicated[:4]  # Cap at 4 to keep outreach generation fast

        logger.warning("All live search sources returned empty — no contacts found.")
    else:
        logger.info("TAVILY_API_KEY not set — live search disabled.")

    return []

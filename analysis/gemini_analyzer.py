"""
Gemini AI Analyzer - Stage 5
Uses Google Gemini to generate neutral, evidence-based analysis from
the structured multi-source comparison produced by Stage 4.

HOW IT WORKS:
=============

1. Takes the comparison result from comparator.py (agreed facts, differences,
   contradictions, unique claims, named entities).

2. Formats this into a structured prompt that instructs Gemini to:
   - Summarize the event neutrally
   - Identify information supported by multiple sources
   - Highlight differences and contradictions
   - Note claims from single sources
   - Preserve source attribution
   - Acknowledge uncertainty

3. Requests JSON-formatted output from Gemini for predictable parsing.

4. Returns a structured analysis dict that downstream code can use.

IMPORTANT DESIGN PRINCIPLES:
   - Never declares a source "fake" just because it differs
   - Distinguishes reported information from verified facts
   - Acknowledges uncertainty when evidence is insufficient
   - Does not invent information beyond what the articles provide
"""

import json
import time

from google import genai
from google.genai import types as genai_types

from config import GEMINI_API_KEY

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Gemini model to use. gemini-3.6-flash is fast and cost-effective.
DEFAULT_MODEL = "gemini-3.6-flash"

# Free tier: 5 requests per minute. We wait 15s between calls to stay safe.
_RATE_LIMIT_DELAY_SECONDS = 15

# Max groups to analyze with Gemini. Prevents rate-limit errors on broad topics.
MAX_GROUPS_TO_ANALYZE = 5

# Expected fields in the Gemini JSON response.
EXPECTED_FIELDS = [
    "summary",
    "agreed_information",
    "differing_information",
    "potential_contradictions",
    "unique_claims",
    "source_assessment",
    "uncertainty_notes",
]


# ──────────────────────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────────────────────

def build_analysis_prompt(comparison_result):
    """
    Build a structured prompt for Gemini from a comparison result.

    Args:
        comparison_result: Dict from comparator.compare_group().

    Returns:
        Prompt string for the Gemini API.
    """
    group_label = comparison_result.get("group_label", "Unknown event")
    sources = comparison_result.get("sources", [])
    article_count = comparison_result.get("article_count", 0)

    # Format agreed information
    agreed = comparison_result.get("agreed_information", [])
    agreed_text = ""
    if agreed:
        items = []
        for item in agreed:
            src_list = ", ".join(item["sources"])
            items.append(f"  - \"{item['fact']}\" (reported by: {src_list})")
        agreed_text = "\n".join(items)
    else:
        agreed_text = "  (none detected)"

    # Format differing information
    differing = comparison_result.get("differing_information", [])
    differing_text = ""
    if differing:
        items = []
        for item in differing:
            versions = "; ".join(
                f"{v['source']}: {v['value']} (\"{v['context']}\")"
                for v in item["versions"]
            )
            items.append(f"  - Topic: {item['topic']} | {versions}")
        differing_text = "\n".join(items)
    else:
        differing_text = "  (none detected)"

    # Format contradictions
    contradictions = comparison_result.get("potential_contradictions", [])
    contradictions_text = ""
    if contradictions:
        items = []
        for item in contradictions:
            claims = "; ".join(
                f"{c['source']}: {c['claim']}" for c in item["claims"]
            )
            items.append(f"  - {item['topic']} | {claims}")
        contradictions_text = "\n".join(items)
    else:
        contradictions_text = "  (none detected)"

    # Format unique information
    unique = comparison_result.get("unique_information", [])
    unique_text = ""
    if unique:
        items = [f"  - \"{item['fact']}\" (only in: {item['source']})" for item in unique]
        unique_text = "\n".join(items)
    else:
        unique_text = "  (none detected)"

    # Format named information
    named = comparison_result.get("named_information", {})
    named_parts = []
    for noun in named.get("proper_nouns", []):
        named_parts.append(f"  - Entity: {noun['entity']} (from: {', '.join(noun['sources'])})")
    for date in named.get("dates", []):
        named_parts.append(f"  - Date: {date['date']} (from: {', '.join(date['sources'])})")
    for num in named.get("numbers", []):
        named_parts.append(f"  - Number: {num['value']} in \"{num['context']}\" ({num['source']})")
    named_text = "\n".join(named_parts) if named_parts else "  (none detected)"

    prompt = f"""You are a neutral news analyst. Analyze the following multi-source news comparison and produce a structured analysis.

EVENT: "{group_label}"
SOURCES: {', '.join(sources)} ({article_count} articles)

=== INFORMATION AGREED BY MULTIPLE SOURCES ===
{agreed_text}

=== DIFFERING INFORMATION BETWEEN SOURCES ===
{differing_text}

=== POTENTIAL CONTRADICTIONS ===
{contradictions_text}

=== UNIQUE INFORMATION (SINGLE SOURCE ONLY) ===
{unique_text}

=== NAMED INFORMATION (ENTITIES, DATES, NUMBERS) ===
{named_text}

INSTRUCTIONS:
1. Summarize the event based ONLY on the information provided above.
2. Identify what multiple sources agree on — these are the most supported facts.
3. Highlight where sources differ and explain the differences neutrally.
4. Flag potential contradictions but do NOT declare any source "fake" or "wrong" just because it differs from others.
5. Note claims that appear in only one source — these are less corroborated but not necessarily false.
6. Provide a brief assessment of each source's contribution.
7. Acknowledge any uncertainty — if the evidence is insufficient to draw a conclusion, say so explicitly.
8. Do NOT invent or assume any information not present in the data above.
9. Distinguish between "reported" information and "verified" facts.

Respond with a JSON object containing these exact fields:
- "summary": A neutral 2-4 sentence summary of the event.
- "agreed_information": A list of strings describing facts supported by multiple sources.
- "differing_information": A list of strings describing where sources disagree, with source names.
- "potential_contradictions": A list of strings describing potential contradictions, or an empty list if none.
- "unique_claims": A list of objects with "claim" and "source" fields for single-source information.
- "source_assessment": A list of objects with "source" and "assessment" fields briefly describing each source's contribution.
- "uncertainty_notes": A list of strings noting areas where more information is needed.

Respond ONLY with a valid JSON object. No explanation outside the JSON."""

    return prompt


# ──────────────────────────────────────────────────────────────
# Gemini API Interaction
# ──────────────────────────────────────────────────────────────

def create_gemini_client(api_key=None):
    """
    Create a Gemini API client.

    Args:
        api_key: Optional API key. Defaults to GEMINI_API_KEY from config.

    Returns:
        genai.Client instance.

    Raises:
        ValueError: If no API key is available.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "Gemini API key is not configured. "
            "Please set GEMINI_API_KEY in your .env file."
        )
    return genai.Client(api_key=key)


def call_gemini(client, prompt, model=DEFAULT_MODEL, _retry=True):
    """
    Send a prompt to Gemini and return the raw response.
    Automatically retries once after a wait if rate-limited (429).

    Args:
        client: genai.Client instance.
        prompt: The prompt string.
        model: Gemini model name (default: gemini-3.6-flash).
        _retry: Internal flag — retries once on rate limit.

    Returns:
        The Gemini response object.

    Raises:
        ConnectionError: On network errors.
        RuntimeError: On API errors (invalid key, rate limit, etc.).
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # Low temperature for factual analysis
            ),
        )
        return response
    except genai.errors.ClientError as e:
        error_msg = str(e)
        # Redact API key from error messages
        if GEMINI_API_KEY and GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(GEMINI_API_KEY, "[REDACTED]")
        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            raise RuntimeError(f"Invalid Gemini API key: {error_msg}") from e
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg.upper():
            if _retry:
                # Wait 60 seconds and retry once
                time.sleep(60)
                return call_gemini(client, prompt, model, _retry=False)
            raise RuntimeError(f"Gemini rate limit exceeded: {error_msg}") from e
        raise RuntimeError(f"Gemini API error: {error_msg}") from e
    except genai.errors.ServerError as e:
        error_msg = str(e)
        if GEMINI_API_KEY and GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(GEMINI_API_KEY, "[REDACTED]")
        raise RuntimeError(f"Gemini server error: {error_msg}") from e
    except Exception as e:
        error_msg = str(e)
        if GEMINI_API_KEY and GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(GEMINI_API_KEY, "[REDACTED]")
        raise ConnectionError(f"Network or connection error: {error_msg}") from e


def parse_gemini_response(response):
    """
    Parse the Gemini response into a structured dict.

    Args:
        response: The Gemini response object.

    Returns:
        Parsed dict with the expected analysis fields.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    try:
        text = response.text
    except (AttributeError, TypeError):
        raise ValueError("Gemini returned an empty or invalid response.")

    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response.")

    # Attempt to parse as JSON
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini response is not valid JSON: {e}. "
            f"Response starts with: {text[:200]}"
        ) from e

    if not isinstance(result, dict):
        raise ValueError(
            f"Gemini response is not a JSON object. Got: {type(result).__name__}"
        )

    # Ensure all expected fields exist with defaults
    defaults = {
        "summary": "No summary available.",
        "agreed_information": [],
        "differing_information": [],
        "potential_contradictions": [],
        "unique_claims": [],
        "source_assessment": [],
        "uncertainty_notes": [],
    }

    for field, default in defaults.items():
        if field not in result:
            result[field] = default

    return result


# ──────────────────────────────────────────────────────────────
# Main Analysis Function
# ──────────────────────────────────────────────────────────────

def analyze_group(comparison_result, client=None, model=DEFAULT_MODEL):
    """
    Analyze a single event group comparison using Gemini.

    Args:
        comparison_result: Dict from comparator.compare_group().
        client: Optional pre-created genai.Client. Created automatically if None.
        model: Gemini model name (default: gemini-2.0-flash).

    Returns:
        Dict with:
            "group_id": int
            "group_label": str
            "model_used": str
            "analysis": the parsed Gemini analysis dict
            "status": "success" or "error"
            "error": error message if status is "error", else None

    Raises:
        Nothing — errors are caught and returned in the result dict.
    """
    group_id = comparison_result.get("group_id", 0)
    group_label = comparison_result.get("group_label", "Unknown")

    # Handle empty input
    if not comparison_result.get("articles", comparison_result.get("article_count", 0)):
        # Check article_count since articles list may not be in comparison result
        if comparison_result.get("article_count", 0) == 0:
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": model,
                "analysis": None,
                "status": "error",
                "error": "No articles in comparison result to analyze.",
            }

    # Build prompt
    prompt = build_analysis_prompt(comparison_result)

    # Create client if not provided
    try:
        if client is None:
            client = create_gemini_client()
    except ValueError as e:
        return {
            "group_id": group_id,
            "group_label": group_label,
            "model_used": model,
            "analysis": None,
            "status": "error",
            "error": str(e),
        }

    # Call Gemini
    try:
        response = call_gemini(client, prompt, model)
    except (RuntimeError, ConnectionError) as e:
        return {
            "group_id": group_id,
            "group_label": group_label,
            "model_used": model,
            "analysis": None,
            "status": "error",
            "error": str(e),
        }

    # Parse response
    try:
        analysis = parse_gemini_response(response)
    except ValueError as e:
        return {
            "group_id": group_id,
            "group_label": group_label,
            "model_used": model,
            "analysis": None,
            "status": "error",
            "error": str(e),
        }

    return {
        "group_id": group_id,
        "group_label": group_label,
        "model_used": model,
        "analysis": analysis,
        "status": "success",
        "error": None,
    }


def analyze_all_groups(comparison_results, client=None, model=DEFAULT_MODEL):
    """
    Analyze all event group comparisons using Gemini.

    Args:
        comparison_results: List of dicts from comparator.compare_all_groups().
        client: Optional pre-created genai.Client.
        model: Gemini model name.

    Returns:
        List of analysis result dicts (one per group).
    """
    if client is None:
        try:
            client = create_gemini_client()
        except ValueError as e:
            return [{
                "group_id": 0,
                "group_label": "N/A",
                "model_used": model,
                "analysis": None,
                "status": "error",
                "error": str(e),
            }]

    # Cap to MAX_GROUPS_TO_ANALYZE to avoid burning through free-tier quota
    # on broad topics (e.g. "cricket" returns 11+ groups)
    groups_to_analyze = comparison_results[:MAX_GROUPS_TO_ANALYZE]

    results = []
    for i, comparison in enumerate(groups_to_analyze):
        result = analyze_group(comparison, client=client, model=model)
        results.append(result)
        # Throttle: wait between calls to respect free-tier rate limit (5 RPM)
        # Skip delay after the last group
        if i < len(groups_to_analyze) - 1:
            time.sleep(_RATE_LIMIT_DELAY_SECONDS)

    return results


# ──────────────────────────────────────────────────────────────
# Report Printing
# ──────────────────────────────────────────────────────────────

def print_analysis_report(result):
    """
    Print a human-readable AI analysis report.

    Args:
        result: Dict returned by analyze_group().
    """
    print(f"\n{'=' * 70}")
    print(f"  AI ANALYSIS: \"{result['group_label']}\"")
    print(f"{'=' * 70}")
    print(f"  Group ID:  {result['group_id']}")
    print(f"  Model:     {result['model_used']}")
    print(f"  Status:    {result['status']}")

    if result["status"] == "error":
        print(f"  Error:     {result['error']}")
        print()
        return

    analysis = result["analysis"]

    # Summary
    print(f"\n  --- SUMMARY ---")
    print(f"  {analysis.get('summary', 'N/A')}")

    # Agreed information
    agreed = analysis.get("agreed_information", [])
    print(f"\n  --- AGREED INFORMATION ({len(agreed)} items) ---")
    for item in agreed:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item.get('fact', item)}")

    # Differing information
    differing = analysis.get("differing_information", [])
    print(f"\n  --- DIFFERING INFORMATION ({len(differing)} items) ---")
    for item in differing:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item}")

    # Contradictions
    contradictions = analysis.get("potential_contradictions", [])
    print(f"\n  --- POTENTIAL CONTRADICTIONS ({len(contradictions)} items) ---")
    for item in contradictions:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item}")

    # Unique claims
    unique = analysis.get("unique_claims", [])
    print(f"\n  --- UNIQUE CLAIMS ({len(unique)} items) ---")
    for item in unique:
        if isinstance(item, dict):
            print(f"    - \"{item.get('claim', 'N/A')}\" (source: {item.get('source', 'N/A')})")
        elif isinstance(item, str):
            print(f"    - {item}")

    # Source assessment
    assessment = analysis.get("source_assessment", [])
    print(f"\n  --- SOURCE ASSESSMENT ({len(assessment)} items) ---")
    for item in assessment:
        if isinstance(item, dict):
            print(f"    - {item.get('source', 'N/A')}: {item.get('assessment', 'N/A')}")
        elif isinstance(item, str):
            print(f"    - {item}")

    # Uncertainty notes
    uncertainty = analysis.get("uncertainty_notes", [])
    print(f"\n  --- UNCERTAINTY NOTES ({len(uncertainty)} items) ---")
    for item in uncertainty:
        if isinstance(item, str):
            print(f"    - {item}")

    print()

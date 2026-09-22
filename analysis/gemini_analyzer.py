"""
AI Analyzer - Multi-Source News Analysis
Uses OpenRouter (or Google Gemini fallback) to generate neutral, evidence-based
analysis from the structured multi-source comparison produced by Stage 4.

HOW IT WORKS:
=============
1. Takes the comparison result from comparator.py (agreed facts, differences,
   contradictions, unique claims, named entities).
2. Formats this into a structured prompt instructing the AI to:
   - Summarize the event neutrally
   - Identify information supported by multiple sources
   - Highlight differences and contradictions
   - Note claims from single sources
   - Preserve source attribution
   - Acknowledge uncertainty
3. Parses JSON-formatted output for predictable, structured reporting.
"""

import json
import re
import time
import requests

from config import OPENROUTER_API_KEY, GEMINI_API_KEY

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Top active free models on OpenRouter (100% free, reliable)
DEFAULT_OPENROUTER_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Max groups to analyze per search
MAX_GROUPS_TO_ANALYZE = 5


# Expected fields in the AI JSON response
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
    Build a structured prompt from a comparison result.

    Args:
        comparison_result: Dict from comparator.compare_group().

    Returns:
        Prompt string for the AI API.
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
                for v in item.get("versions", [])
            )
            items.append(f"  - Topic: {item.get('topic', 'General')} | {versions}")
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
                f"{c['source']}: {c['claim']}" for c in item.get("claims", [])
            )
            items.append(f"  - {item.get('topic', 'Topic')} | {claims}")
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

Respond ONLY with a JSON object containing these exact fields:
{{
  "summary": "A neutral 2-4 sentence summary of the event.",
  "agreed_information": ["fact 1 supported by multiple sources", "fact 2..."],
  "differing_information": ["differing point 1 with source names", "differing point 2..."],
  "potential_contradictions": ["potential contradiction 1", "or empty list"],
  "unique_claims": [{{"claim": "single-source fact", "source": "Source Name"}}],
  "source_assessment": [{{"source": "Source Name", "assessment": "brief neutral evaluation"}}],
  "uncertainty_notes": ["area of uncertainty 1", "area 2..."]
}}

Respond ONLY with valid JSON. No markdown code blocks around it, no commentary."""

    return prompt


# ──────────────────────────────────────────────────────────────
# API Interaction (OpenRouter with Gemini Fallback)
# ──────────────────────────────────────────────────────────────

def call_openrouter(prompt, api_key=None, model=DEFAULT_MODEL):
    """
    Call OpenRouter API using requests.
    Attempts primary model and automatically falls back if needed.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key is not configured.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vaibhavjadhav0391-wq/news",
        "X-Title": "News Analyzer"
    }

    models_to_try = [model] + [m for m in DEFAULT_OPENROUTER_MODELS if m != model]

    last_error = None
    for target_model in models_to_try:
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional neutral news analyst. Always respond in valid raw JSON."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                return content, target_model
            else:
                last_error = f"Status {res.status_code}: {res.text[:200]}"
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"OpenRouter API call failed on all models: {last_error}")


def create_gemini_client(api_key=None):
    """Fallback client for legacy Gemini API."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Neither OpenRouter nor Gemini API key is configured.")
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as e:
        return None


def call_gemini(client, prompt, model="gemini-3.6-flash"):
    """Fallback Gemini API call."""
    if client is None:
        raise ValueError("Gemini client is not initialized.")
    from google.genai import types as genai_types
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return response.text


def parse_ai_response(text):
    """
    Parse the AI response string into a structured dict.
    Extracts JSON even if wrapped in markdown blocks.
    """
    if not text or not text.strip():
        raise ValueError("AI returned an empty response.")

    cleaned = text.strip()
    # If wrapped in ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Find first { and last }
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AI response is not valid JSON: {e}. Raw: {text[:200]}"
        ) from e

    if not isinstance(result, dict):
        raise ValueError(f"AI response is not a JSON object. Got: {type(result).__name__}")

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


parse_gemini_response = parse_ai_response


# ──────────────────────────────────────────────────────────────
# Main Analysis Function
# ──────────────────────────────────────────────────────────────

def analyze_group(comparison_result, client=None, model=DEFAULT_MODEL):
    """
    Analyze a single event group comparison using OpenRouter (or Gemini fallback).

    Returns:
        Dict with group_id, group_label, model_used, analysis, status, error.
    """
    group_id = comparison_result.get("group_id", 0)
    group_label = comparison_result.get("group_label", "Unknown")

    if not comparison_result.get("articles", comparison_result.get("article_count", 0)):
        if comparison_result.get("article_count", 0) == 0:
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": model,
                "analysis": None,
                "status": "error",
                "error": "No articles in comparison result to analyze.",
            }

    prompt = build_analysis_prompt(comparison_result)

    # 1. Primary: Try OpenRouter
    if OPENROUTER_API_KEY:
        try:
            raw_text, used_model = call_openrouter(prompt, api_key=OPENROUTER_API_KEY, model=model)
            analysis = parse_ai_response(raw_text)
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": used_model,
                "analysis": analysis,
                "status": "success",
                "error": None,
            }
        except Exception as e:
            # Fall through to Gemini if available
            pass

    # 2. Fallback: Try Gemini
    if GEMINI_API_KEY:
        try:
            if client is None:
                client = create_gemini_client()
            raw_text = call_gemini(client, prompt)
            analysis = parse_ai_response(raw_text)
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": "gemini-3.6-flash",
                "analysis": analysis,
                "status": "success",
                "error": None,
            }
        except Exception as e:
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
        "analysis": None,
        "status": "error",
        "error": "No valid OpenRouter or Gemini API key configured.",
    }


def analyze_all_groups(comparison_results, client=None, model=DEFAULT_MODEL):
    """
    Analyze all event group comparisons.
    With OpenRouter, fast analysis is performed without 13-second delays!
    """
    groups_to_analyze = comparison_results[:MAX_GROUPS_TO_ANALYZE]
    results = []
    for comparison in groups_to_analyze:
        result = analyze_group(comparison, client=client, model=model)
        results.append(result)
        # Small 0.5s pause to prevent local network flood
        time.sleep(0.5)

    return results


# ──────────────────────────────────────────────────────────────
# Report Printing
# ──────────────────────────────────────────────────────────────

def print_analysis_report(result):
    """
    Print a human-readable AI analysis report.
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
    print(f"\n  --- SUMMARY ---")
    print(f"  {analysis.get('summary', 'N/A')}")

    agreed = analysis.get("agreed_information", [])
    print(f"\n  --- AGREED INFORMATION ({len(agreed)} items) ---")
    for item in agreed:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item.get('fact', item)}")

    differing = analysis.get("differing_information", [])
    print(f"\n  --- DIFFERING INFORMATION ({len(differing)} items) ---")
    for item in differing:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item}")

    contradictions = analysis.get("potential_contradictions", [])
    print(f"\n  --- POTENTIAL CONTRADICTIONS ({len(contradictions)} items) ---")
    for item in contradictions:
        if isinstance(item, str):
            print(f"    - {item}")
        elif isinstance(item, dict):
            print(f"    - {item}")

    unique = analysis.get("unique_claims", [])
    print(f"\n  --- UNIQUE CLAIMS ({len(unique)} items) ---")
    for item in unique:
        if isinstance(item, dict):
            print(f"    - \"{item.get('claim', 'N/A')}\" (source: {item.get('source', 'N/A')})")
        elif isinstance(item, str):
            print(f"    - {item}")

    assessment = analysis.get("source_assessment", [])
    print(f"\n  --- SOURCE ASSESSMENT ({len(assessment)} items) ---")
    for item in assessment:
        if isinstance(item, dict):
            print(f"    - {item.get('source', 'N/A')}: {item.get('assessment', 'N/A')}")
        elif isinstance(item, str):
            print(f"    - {item}")

    uncertainty = analysis.get("uncertainty_notes", [])
    print(f"\n  --- UNCERTAINTY NOTES ({len(uncertainty)} items) ---")
    for item in uncertainty:
        if isinstance(item, str):
            print(f"    - {item}")

    print()

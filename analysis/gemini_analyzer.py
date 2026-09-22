"""
AI Analyzer - Multi-Source News Analysis
Uses Groq (Ultra-fast 0.6s Llama/GPT models), with OpenRouter & Google Gemini fallbacks.
Produces structured, neutral, evidence-based comparisons.

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
3. Executes multi-threaded parallel requests for near-instant responses.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from config import GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Ultra-fast Groq models (0.5s - 1.2s response time)
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
]

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Max groups to analyze per search (top 3-4 most prominent multi-source events)
MAX_GROUPS_TO_ANALYZE = 4

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

Respond with a JSON object containing these exact fields:
{{
  "summary": "A neutral 2-4 sentence summary of the event.",
  "agreed_information": ["fact 1 supported by multiple sources", "fact 2..."],
  "differing_information": ["differing point 1 with source names", "differing point 2..."],
  "potential_contradictions": ["potential contradiction 1", "or empty list"],
  "unique_claims": [{{"claim": "single-source fact", "source": "Source Name"}}],
  "source_assessment": [{{"source": "Source Name", "assessment": "brief neutral evaluation"}}],
  "uncertainty_notes": ["area of uncertainty 1", "area 2..."]
}}

Respond ONLY with valid raw JSON."""

    return prompt


# ──────────────────────────────────────────────────────────────
# API Engine: Groq (Primary) -> OpenRouter -> Gemini (Fallbacks)
# ──────────────────────────────────────────────────────────────

def call_groq(prompt, api_key=None, model=None):
    """
    Call Groq API (Blazing fast, 0.5-1.5s latency).
    """
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("Groq API key is not configured.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    models_to_try = [model] if model in GROQ_MODELS else GROQ_MODELS

    last_error = None
    for target_model in models_to_try:
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional neutral news analyst. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12
            )
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                return content, f"groq/{target_model}"
            else:
                last_error = f"Status {res.status_code}: {res.text[:200]}"
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"Groq API call failed: {last_error}")


def call_openrouter(prompt, api_key=None):
    """Call OpenRouter API as secondary fallback."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key is not configured.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vaibhavjadhav0391-wq/news",
        "X-Title": "News Analyzer"
    }

    models = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ]

    last_error = None
    for target_model in models:
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": "You are a neutral news analyst. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"], target_model
            last_error = res.text[:200]
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"OpenRouter failed: {last_error}")


def create_gemini_client(api_key=None):
    """Fallback client for legacy Gemini API."""
    key = api_key or GEMINI_API_KEY
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
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
    """
    if not text or not text.strip():
        raise ValueError("AI returned an empty response.")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI response is not valid JSON: {e}. Raw: {text[:200]}") from e

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
# Main Analysis Functions
# ──────────────────────────────────────────────────────────────

def analyze_group(comparison_result, client=None, model=DEFAULT_MODEL):
    """
    Analyze a single event group comparison using the fastest available engine.
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

    # 1. Primary: Groq (Blazing fast 0.6s)
    if GROQ_API_KEY:
        try:
            raw_text, used_model = call_groq(prompt, api_key=GROQ_API_KEY, model=model)
            analysis = parse_ai_response(raw_text)
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": used_model,
                "analysis": analysis,
                "status": "success",
                "error": None,
            }
        except Exception:
            pass

    # 2. Secondary: OpenRouter
    if OPENROUTER_API_KEY:
        try:
            raw_text, used_model = call_openrouter(prompt, api_key=OPENROUTER_API_KEY)
            analysis = parse_ai_response(raw_text)
            return {
                "group_id": group_id,
                "group_label": group_label,
                "model_used": used_model,
                "analysis": analysis,
                "status": "success",
                "error": None,
            }
        except Exception:
            pass

    # 3. Tertiary: Gemini
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
        "error": "No valid AI API key configured.",
    }


def analyze_all_groups(comparison_results, client=None, model=DEFAULT_MODEL):
    """
    Analyze all event group comparisons in parallel for instant speed.
    """
    sorted_groups = sorted(
        comparison_results,
        key=lambda g: g.get("article_count", len(g.get("sources", []))),
        reverse=True
    )
    groups_to_analyze = sorted_groups[:MAX_GROUPS_TO_ANALYZE]

    if not groups_to_analyze:
        return []

    results_dict = {}
    with ThreadPoolExecutor(max_workers=min(len(groups_to_analyze), 4)) as executor:
        future_to_idx = {
            executor.submit(analyze_group, comp, client, model): i
            for i, comp in enumerate(groups_to_analyze)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_dict[idx] = future.result()
            except Exception as e:
                comp = groups_to_analyze[idx]
                results_dict[idx] = {
                    "group_id": comp.get("group_id", idx + 1),
                    "group_label": comp.get("group_label", "Unknown"),
                    "model_used": model,
                    "analysis": None,
                    "status": "error",
                    "error": str(e),
                }

    return [results_dict[i] for i in range(len(groups_to_analyze))]


def print_analysis_report(result):
    """Print formatted analysis report."""
    print(f"\n{'=' * 70}")
    print(f"  AI ANALYSIS: \"{result['group_label']}\"")
    print(f"{'=' * 70}")
    print(f"  Group ID:  {result['group_id']}")
    print(f"  Model:     {result['model_used']}")
    print(f"  Status:    {result['status']}")

    if result["status"] == "error":
        print(f"  Error:     {result['error']}\n")
        return

    analysis = result["analysis"]
    print(f"\n  --- SUMMARY ---\n  {analysis.get('summary', 'N/A')}")
    print()

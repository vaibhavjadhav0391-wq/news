"""
Test script for the Gemini AI Analyzer.

All tests use MOCKED Gemini API responses — no real API calls are made.
This ensures tests are fast, free, and repeatable.
"""

import json
from unittest.mock import MagicMock, patch

from analysis.gemini_analyzer import (
    build_analysis_prompt,
    create_gemini_client,
    parse_gemini_response,
    analyze_group,
    analyze_all_groups,
    call_gemini,
    EXPECTED_FIELDS,
    DEFAULT_MODEL,
)

# ──────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────
passed = 0
failed = 0


def check(label, condition):
    """Simple test assertion with pass/fail tracking."""
    global passed, failed
    if condition:
        print(f"  PASS: {label}")
        passed += 1
    else:
        print(f"  FAIL: {label}")
        failed += 1


def make_comparison(group_id=1, label="Test Event", sources=None,
                    article_count=2, agreed=None, differing=None,
                    contradictions=None, unique=None, named=None):
    """Helper to create a comparison result dict like comparator.py produces."""
    return {
        "group_id": group_id,
        "group_label": label,
        "sources": sources or ["Source A", "Source B"],
        "article_count": article_count,
        "agreed_information": agreed or [
            {"fact": "earthquake", "sources": ["Source A", "Source B"], "source_count": 2},
        ],
        "differing_information": differing or [],
        "potential_contradictions": contradictions or [],
        "unique_information": unique or [],
        "named_information": named or {
            "proper_nouns": [],
            "dates": [],
            "numbers": [],
        },
    }


def make_valid_gemini_json():
    """Return a valid Gemini analysis JSON string."""
    return json.dumps({
        "summary": "A significant earthquake struck the region, causing widespread damage.",
        "agreed_information": [
            "An earthquake occurred in the region",
            "Multiple casualties were reported",
        ],
        "differing_information": [
            "Source A reports 50 casualties while Source B reports 75",
        ],
        "potential_contradictions": [
            "Source A says the government confirmed aid, Source B says aid was denied",
        ],
        "unique_claims": [
            {"claim": "International rescue teams deployed", "source": "Source A"},
            {"claim": "Aftershock warning issued", "source": "Source B"},
        ],
        "source_assessment": [
            {"source": "Source A", "assessment": "Provides detailed casualty figures"},
            {"source": "Source B", "assessment": "Focuses on government response"},
        ],
        "uncertainty_notes": [
            "Exact casualty figures cannot be confirmed from available sources",
        ],
    })


def make_mock_response(text):
    """Create a mock Gemini response object with a .text attribute."""
    mock = MagicMock()
    mock.text = text
    return mock


# ──────────────────────────────────────────────────────────────
# Test 1: Prompt construction
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Prompt Construction")
print("=" * 70)

comparison = make_comparison(
    label="Earthquake in Turkey",
    sources=["BBC", "Al Jazeera"],
    agreed=[
        {"fact": "earthquake", "sources": ["BBC", "Al Jazeera"], "source_count": 2},
        {"fact": "turkey", "sources": ["BBC", "Al Jazeera"], "source_count": 2},
    ],
    unique=[
        {"fact": "magnitude 7.1", "source": "BBC"},
    ],
)

prompt = build_analysis_prompt(comparison)

check("Prompt is a string", isinstance(prompt, str))
check("Prompt contains event label", "Earthquake in Turkey" in prompt)
check("Prompt contains sources", "BBC" in prompt and "Al Jazeera" in prompt)
check("Prompt contains agreed info", "earthquake" in prompt)
check("Prompt contains unique info", "magnitude 7.1" in prompt)
check("Prompt asks for JSON", "JSON" in prompt)
check("Prompt says do not declare fake", "fake" in prompt.lower())
check("Prompt says acknowledge uncertainty", "uncertainty" in prompt.lower())
check("Prompt says do not invent", "invent" in prompt.lower() or "assume" in prompt.lower())
check("Prompt mentions source attribution", "attribution" in prompt.lower() or "source" in prompt.lower())


# ──────────────────────────────────────────────────────────────
# Test 2: Valid JSON response parsing
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: Valid JSON Response Parsing")
print("=" * 70)

valid_response = make_mock_response(make_valid_gemini_json())
result = parse_gemini_response(valid_response)

check("Result is a dict", isinstance(result, dict))
for field in EXPECTED_FIELDS:
    check(f"Has field '{field}'", field in result)

check("Summary is a string", isinstance(result["summary"], str))
check("Summary is non-empty", len(result["summary"]) > 0)
check("Agreed info is a list", isinstance(result["agreed_information"], list))
check("Unique claims is a list", isinstance(result["unique_claims"], list))
check("Source assessment is a list", isinstance(result["source_assessment"], list))
check("Uncertainty notes is a list", isinstance(result["uncertainty_notes"], list))


# ──────────────────────────────────────────────────────────────
# Test 3: Multiple sources - agreed information
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 3: Multiple Sources with Agreed Information")
print("=" * 70)

multi_source = make_comparison(
    sources=["Reuters", "BBC", "AP"],
    article_count=3,
    agreed=[
        {"fact": "election", "sources": ["Reuters", "BBC", "AP"], "source_count": 3},
        {"fact": "results", "sources": ["Reuters", "BBC"], "source_count": 2},
    ],
)

prompt = build_analysis_prompt(multi_source)
check("Prompt includes all 3 sources", all(s in prompt for s in ["Reuters", "BBC", "AP"]))
check("Prompt includes agreed facts", "election" in prompt and "results" in prompt)

# Simulate Gemini response
mock_response = make_mock_response(json.dumps({
    "summary": "Election results announced across multiple sources.",
    "agreed_information": ["Election results were announced", "Multiple sources confirm outcome"],
    "differing_information": [],
    "potential_contradictions": [],
    "unique_claims": [],
    "source_assessment": [
        {"source": "Reuters", "assessment": "Wire service coverage"},
        {"source": "BBC", "assessment": "Detailed analysis"},
        {"source": "AP", "assessment": "Breaking news focus"},
    ],
    "uncertainty_notes": [],
}))

parsed = parse_gemini_response(mock_response)
check("Agreed information has 2 items", len(parsed["agreed_information"]) == 2)
check("Source assessment has 3 items", len(parsed["source_assessment"]) == 3)


# ──────────────────────────────────────────────────────────────
# Test 4: Differing information
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 4: Differing Information")
print("=" * 70)

differing_comp = make_comparison(
    differing=[{
        "topic": "casualties",
        "versions": [
            {"source": "BBC", "value": "50", "context": "killed 50 people"},
            {"source": "Al Jazeera", "value": "75", "context": "killed 75 people"},
        ],
    }],
)

prompt = build_analysis_prompt(differing_comp)
check("Prompt contains differing info", "50" in prompt and "75" in prompt)
check("Prompt contains source attribution for differences",
      "BBC" in prompt and "Al Jazeera" in prompt)


# ──────────────────────────────────────────────────────────────
# Test 5: Contradictions
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Potential Contradictions")
print("=" * 70)

contradiction_comp = make_comparison(
    contradictions=[{
        "topic": "Opposing language (confirmed/denied)",
        "claims": [
            {"source": "State News", "claim": "Uses: confirmed"},
            {"source": "Independent", "claim": "Uses: denied"},
        ],
    }],
)

prompt = build_analysis_prompt(contradiction_comp)
check("Prompt contains contradiction info", "confirmed" in prompt and "denied" in prompt)


# ──────────────────────────────────────────────────────────────
# Test 6: Unique claims
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 6: Unique Claims")
print("=" * 70)

unique_comp = make_comparison(
    unique=[
        {"fact": "tsunami warning", "source": "Source A"},
        {"fact": "evacuation ordered", "source": "Source B"},
    ],
)

prompt = build_analysis_prompt(unique_comp)
check("Prompt contains unique claims", "tsunami warning" in prompt)
check("Unique claims attributed to sources", "Source A" in prompt)

# Simulate response with unique claims
mock_response = make_mock_response(json.dumps({
    "summary": "Event summary.",
    "agreed_information": [],
    "differing_information": [],
    "potential_contradictions": [],
    "unique_claims": [
        {"claim": "Tsunami warning issued", "source": "Source A"},
        {"claim": "Evacuation ordered", "source": "Source B"},
    ],
    "source_assessment": [],
    "uncertainty_notes": ["Tsunami warning only reported by one source"],
}))

parsed = parse_gemini_response(mock_response)
check("Unique claims parsed correctly", len(parsed["unique_claims"]) == 2)
check("Each unique claim has source",
      all("source" in c for c in parsed["unique_claims"]))


# ──────────────────────────────────────────────────────────────
# Test 7: Empty comparison input
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Empty Comparison Input")
print("=" * 70)

empty_comparison = {
    "group_id": 1,
    "group_label": "Empty",
    "sources": [],
    "article_count": 0,
    "agreed_information": [],
    "differing_information": [],
    "potential_contradictions": [],
    "unique_information": [],
    "named_information": {"proper_nouns": [], "dates": [], "numbers": []},
}

# Mock the client to ensure no API call is made
result = analyze_group(empty_comparison, client=MagicMock())
check("Empty input returns error status", result["status"] == "error")
check("Error message mentions no articles", "no article" in result["error"].lower())
check("Analysis is None", result["analysis"] is None)


# ──────────────────────────────────────────────────────────────
# Test 8: Missing API key
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Missing API Key")
print("=" * 70)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", None):
    try:
        create_gemini_client()
        check("Should have raised ValueError", False)
    except ValueError as e:
        check("ValueError raised for missing key", True)
        check("Error mentions API key", "api key" in str(e).lower())
        check("Actual key not in error message",
              "AIza" not in str(e) and "your_actual" not in str(e))

# Also test via analyze_group
with patch("analysis.gemini_analyzer.GEMINI_API_KEY", None):
    result = analyze_group(make_comparison())
    check("Missing key -> error status", result["status"] == "error")
    check("Missing key -> error message", "api key" in result["error"].lower())


# ──────────────────────────────────────────────────────────────
# Test 9: API error handling
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 9: API Error Handling")
print("=" * 70)

# Simulate API error via mock client
mock_client = MagicMock()
mock_client.models.generate_content.side_effect = Exception("Connection refused")

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = analyze_group(make_comparison(), client=mock_client)
    check("API error -> error status", result["status"] == "error")
    check("API error -> has error message", result["error"] is not None)
    check("API error -> analysis is None", result["analysis"] is None)

# Simulate rate limit error
from google.genai import errors as genai_errors

mock_client2 = MagicMock()
mock_client2.models.generate_content.side_effect = Exception("429 RATE_LIMIT exceeded")

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = analyze_group(make_comparison(), client=mock_client2)
    check("Rate limit -> error status", result["status"] == "error")
    check("Rate limit -> error message present", len(result["error"]) > 0)


# ──────────────────────────────────────────────────────────────
# Test 10: Malformed Gemini response
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 10: Malformed Gemini Response")
print("=" * 70)

# Not JSON at all
bad_response1 = make_mock_response("This is not JSON at all.")
try:
    parse_gemini_response(bad_response1)
    check("Should have raised ValueError for non-JSON", False)
except ValueError as e:
    check("ValueError for non-JSON response", True)
    check("Error mentions JSON", "json" in str(e).lower())

# Empty response
empty_response = make_mock_response("")
try:
    parse_gemini_response(empty_response)
    check("Should have raised ValueError for empty", False)
except ValueError as e:
    check("ValueError for empty response", True)
    check("Error mentions empty", "empty" in str(e).lower())

# None text attribute
null_response = MagicMock()
null_response.text = None
try:
    parse_gemini_response(null_response)
    check("Should have raised ValueError for None", False)
except ValueError as e:
    check("ValueError for None response", True)

# JSON array instead of object
array_response = make_mock_response('[1, 2, 3]')
try:
    parse_gemini_response(array_response)
    check("Should have raised ValueError for array", False)
except ValueError as e:
    check("ValueError for JSON array", True)
    check("Error mentions not an object", "object" in str(e).lower())

# Valid JSON but missing fields (should fill defaults)
partial_response = make_mock_response('{"summary": "Partial response"}')
result = parse_gemini_response(partial_response)
check("Partial JSON fills missing fields", all(f in result for f in EXPECTED_FIELDS))
check("Summary preserved from partial", result["summary"] == "Partial response")
check("Missing agreed_information defaults to list",
      isinstance(result["agreed_information"], list) and len(result["agreed_information"]) == 0)


# ──────────────────────────────────────────────────────────────
# Test 11: Full analyze_group with mocked Gemini
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 11: Full analyze_group with Mocked API")
print("=" * 70)

mock_client = MagicMock()
mock_client.models.generate_content.return_value = make_mock_response(
    make_valid_gemini_json()
)

comparison = make_comparison(
    group_id=42,
    label="Test Earthquake",
    sources=["BBC", "Al Jazeera"],
)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = analyze_group(comparison, client=mock_client)

check("Status is success", result["status"] == "success")
check("Error is None", result["error"] is None)
check("Group ID preserved", result["group_id"] == 42)
check("Group label preserved", result["group_label"] == "Test Earthquake")
check("Model is recorded", result["model_used"] == DEFAULT_MODEL)
check("Analysis is a dict", isinstance(result["analysis"], dict))
check("Analysis has summary", "summary" in result["analysis"])
check("Analysis has all expected fields",
      all(f in result["analysis"] for f in EXPECTED_FIELDS))
check("Gemini was called once", mock_client.models.generate_content.call_count == 1)


# ──────────────────────────────────────────────────────────────
# Test 12: analyze_all_groups with mocked Gemini
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 12: analyze_all_groups with Mocked API")
print("=" * 70)

mock_client = MagicMock()
mock_client.models.generate_content.return_value = make_mock_response(
    make_valid_gemini_json()
)

comparisons = [
    make_comparison(group_id=1, label="Event A"),
    make_comparison(group_id=2, label="Event B"),
]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    results = analyze_all_groups(comparisons, client=mock_client)

check("Returns 2 results", len(results) == 2)
check("Both succeeded", all(r["status"] == "success" for r in results))
check("Group IDs preserved", results[0]["group_id"] == 1 and results[1]["group_id"] == 2)
check("Gemini called twice (once per group)",
      mock_client.models.generate_content.call_count == 2)


# ──────────────────────────────────────────────────────────────
# Test 13: Output structure validation
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 13: Output Structure Validation")
print("=" * 70)

mock_client = MagicMock()
mock_client.models.generate_content.return_value = make_mock_response(
    make_valid_gemini_json()
)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = analyze_group(make_comparison(), client=mock_client)

check("Has group_id (int)", isinstance(result["group_id"], int))
check("Has group_label (str)", isinstance(result["group_label"], str))
check("Has model_used (str)", isinstance(result["model_used"], str))
check("Has status (str)", isinstance(result["status"], str))
check("Has analysis (dict)", isinstance(result["analysis"], dict))
check("Has error (None on success)", result["error"] is None)

analysis = result["analysis"]
check("summary is str", isinstance(analysis["summary"], str))
check("agreed_information is list", isinstance(analysis["agreed_information"], list))
check("differing_information is list", isinstance(analysis["differing_information"], list))
check("potential_contradictions is list", isinstance(analysis["potential_contradictions"], list))
check("unique_claims is list", isinstance(analysis["unique_claims"], list))
check("source_assessment is list", isinstance(analysis["source_assessment"], list))
check("uncertainty_notes is list", isinstance(analysis["uncertainty_notes"], list))


# ──────────────────────────────────────────────────────────────
# Test 14: API key not exposed in errors
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 14: API Key Not Exposed in Errors")
print("=" * 70)

fake_key = "AIzaSyDFAKEKEY12345678901234567890"

mock_client = MagicMock()
mock_client.models.generate_content.side_effect = Exception(
    f"Request failed with key {fake_key}"
)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", fake_key):
    result = analyze_group(make_comparison(), client=mock_client)
    check("Error does not contain raw API key",
          fake_key not in result["error"])


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")

"""
Test script for the Flask web application.

All tests use MOCKED pipeline calls -- no real NewsAPI or Gemini requests.
Uses Flask's built-in test client.
"""

import json
from unittest.mock import patch, MagicMock

from app import app

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


def make_success_result(topic="earthquake"):
    """Return a mock pipeline success result."""
    return {
        "status": "success",
        "failed_stage": None,
        "topic": topic,
        "articles_fetched": 5,
        "articles_filtered_out": 1,
        "articles_after_relevance_filter": 4,
        "duplicates_removed": 1,
        "unique_articles": 4,
        "event_groups": 2,
        "groups": [
            {
                "group_id": 1,
                "label": "Test Event",
                "article_count": 2,
                "articles": [
                    {
                        "title": "Earthquake strikes eastern Turkey",
                        "source": {"name": "BBC News"},
                        "author": "Jane Reporter",
                        "publishedAt": "2026-09-01T08:00:00Z",
                        "description": "A powerful earthquake hit eastern Turkey.",
                        "url": "https://bbc.com/news/earthquake-turkey",
                    },
                    {
                        "title": "Turkey earthquake causes damage",
                        "source": {"name": "Reuters"},
                        "author": None,
                        "publishedAt": None,
                        "description": None,
                        "url": "https://reuters.com/earthquake",
                    },
                ],
            },
        ],
        "comparisons": [{"group_id": 1, "group_label": "Test Event"}],
        "analyses": [
            {
                "group_id": 1,
                "group_label": "Test Event",
                "model_used": "gemini-3.6-flash",
                "status": "success",
                "error": None,
                "analysis": {
                    "summary": "Test summary.",
                    "agreed_information": ["Fact A"],
                    "differing_information": [],
                    "potential_contradictions": [],
                    "unique_claims": [],
                    "source_assessment": [],
                    "uncertainty_notes": [],
                },
            }
        ],
        "error": None,
    }


def make_zero_relevant_result(topic="unrelatedtopic"):
    """Return a mock pipeline zero-relevant-articles error result."""
    return {
        "status": "error",
        "failed_stage": "relevance_filter",
        "topic": topic,
        "articles_fetched": 8,
        "articles_filtered_out": 8,
        "articles_after_relevance_filter": 0,
        "duplicates_removed": 0,
        "unique_articles": 0,
        "event_groups": 0,
        "groups": [],
        "comparisons": [],
        "analyses": [],
        "error": "No articles relevant to the query were found for \"unrelatedtopic\". Try a different or broader topic.",
    }



def make_error_result(topic="badtopic"):
    """Return a mock pipeline error result."""
    return {
        "status": "error",
        "failed_stage": "fetch",
        "topic": topic,
        "articles_fetched": 0,
        "duplicates_removed": 0,
        "unique_articles": 0,
        "event_groups": 0,
        "groups": [],
        "comparisons": [],
        "analyses": [],
        "error": "No articles found for \"badtopic\". Try a different topic.",
    }


def make_partial_result(topic="earthquake"):
    """Return a mock pipeline partial result (Gemini failed)."""
    return {
        "status": "partial",
        "failed_stage": "gemini",
        "topic": topic,
        "articles_fetched": 3,
        "duplicates_removed": 0,
        "unique_articles": 3,
        "event_groups": 1,
        "groups": [
            {
                "group_id": 1,
                "label": "Test Event",
                "article_count": 1,
                "articles": [
                    {
                        "title": "Partial result article",
                        "source": {"name": "AP News"},
                        "url": "https://apnews.com/partial",
                    },
                ],
            },
        ],
        "comparisons": [{"group_id": 1, "group_label": "Test Event"}],
        "analyses": [],
        "error": "Gemini analysis failed for all groups.",
    }


# Create a test client
client = app.test_client()
app.config["TESTING"] = True


# ──────────────────────────────────────────────────────────────
# Test 1: Home page loads (GET /)
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Home Page Loads")
print("=" * 70)

response = client.get("/")
html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Contains title 'NEWS ANALYZER'", "NEWS ANALYZER" in html)
check("Contains description text", "multiple sources" in html.lower() or "multi-source" in html.lower())
check("Contains topic input", 'name="topic"' in html)
check("Contains analyze button", "Analyze" in html)
check("Input has id", 'id="topic-input"' in html)
check("Button has id", 'id="analyze-btn"' in html)
check("No error shown on GET", 'id="error-message"' not in html)
check("No result shown on GET", 'id="result-output"' not in html)


# ──────────────────────────────────────────────────────────────
# Test 2: Empty input submission
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: Empty Input Submission")
print("=" * 70)

response = client.post("/", data={"topic": ""})
html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Error message shown", 'id="error-message"' in html)
check("Error mentions topic", "topic" in html.lower() or "enter" in html.lower())
check("No result shown", 'id="result-output"' not in html)

# Whitespace-only input
response2 = client.post("/", data={"topic": "   "})
html2 = response2.data.decode("utf-8")
check("Whitespace -> error shown", 'id="error-message"' in html2)


# ──────────────────────────────────────────────────────────────
# Test 3: Valid topic - pipeline success
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 3: Valid Topic - Pipeline Success")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_success_result()):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Result displayed", 'id="result-output"' in html)
check("Status bar shown", 'id="status-bar"' in html)
check("Status shows SUCCESS", "SUCCESS" in html)
check("No error box", 'id="error-message"' not in html)
check("Topic preserved in input", 'value="earthquake"' in html)
check("Articles count shown", "5" in html)
check("JSON contains summary", "Test summary" in html)


# ──────────────────────────────────────────────────────────────
# Test 4: Pipeline error (no articles)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 4: Pipeline Error")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_error_result()):
    response = client.post("/", data={"topic": "badtopic"})
    html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Error message shown", 'id="error-message"' in html)
check("Error text present",
      "no articles" in html.lower() or "error" in html.lower())


# ──────────────────────────────────────────────────────────────
# Test 5: Pipeline partial success (Gemini failed)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Partial Success (Gemini Failed)")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_partial_result()):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Status bar shown", 'id="status-bar"' in html)
check("Status shows PARTIAL", "PARTIAL" in html)
check("Result still displayed", 'id="result-output"' in html)


# ──────────────────────────────────────────────────────────────
# Test 6: Pipeline exception handling
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 6: Pipeline Exception Handling")
print("=" * 70)

with patch("app.run_pipeline", side_effect=Exception("Unexpected crash")):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("Status code 200 (graceful handling)", response.status_code == 200)
check("Error message shown", 'id="error-message"' in html)
check("Error mentions pipeline",
      "pipeline" in html.lower() or "error" in html.lower())
check("No result output", 'id="result-output"' not in html)


# ──────────────────────────────────────────────────────────────
# Test 7: Result JSON rendering
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Result JSON Rendering")
print("=" * 70)

success = make_success_result()
with patch("app.run_pipeline", return_value=success):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("JSON output block present", 'id="result-output"' in html)
# Jinja2 HTML-escapes quotes: " becomes &#34; in rendered output
check("JSON contains status field",
      "status" in html and ("success" in html.lower() or "&#34;status&#34;" in html))
check("JSON contains topic",
      "earthquake" in html)
check("JSON contains analyses",
      "analyses" in html)
check("JSON contains comparisons",
      "comparisons" in html)


# ──────────────────────────────────────────────────────────────
# Test 8: Semantic HTML checks
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Semantic HTML Checks")
print("=" * 70)

response = client.get("/")
html = response.data.decode("utf-8")

check("Has DOCTYPE", "<!DOCTYPE html>" in html)
check("Has lang attribute", 'lang="en"' in html)
check("Has meta charset", 'charset="UTF-8"' in html)
check("Has meta viewport", "viewport" in html)
check("Has meta description", 'meta name="description"' in html)
check("Has title tag", "<title>" in html)
check("Has h1 heading", "<h1" in html)
check("Has form element", "<form" in html)
check("Form method is POST", 'method="POST"' in html)


# ──────────────────────────────────────────────────────────────
# Test 9: API key not exposed in responses
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 9: API Key Not Exposed")
print("=" * 70)

# Check GET response
response = client.get("/")
html = response.data.decode("utf-8")
check("No NEWS_API_KEY in GET response",
      "NEWS_API_KEY" not in html)
check("No GEMINI_API_KEY in GET response",
      "GEMINI_API_KEY" not in html)

# Check POST response with result
with patch("app.run_pipeline", return_value=make_success_result()):
    response = client.post("/", data={"topic": "test"})
    html = response.data.decode("utf-8")
    check("No NEWS_API_KEY in POST response",
          "NEWS_API_KEY" not in html)
    check("No GEMINI_API_KEY in POST response",
          "GEMINI_API_KEY" not in html)


# ──────────────────────────────────────────────────────────────
# Test 10: Source article display
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 10: Source Article Display")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_success_result()):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

# Article title is displayed
check("Article title displayed",
      "Earthquake strikes eastern Turkey" in html)

# Source/publication name is displayed
check("Source name displayed", "BBC News" in html)

# Author displayed when available
check("Author displayed when present", "Jane Reporter" in html)

# Published date displayed when available
check("Published date displayed", "2026-09-01" in html)

# Description displayed when available
check("Description displayed",
      "powerful earthquake" in html.lower())

# Original URL rendered as clickable link
check("URL is a clickable link",
      'href="https://bbc.com/news/earthquake-turkey"' in html)
check("Link opens in new tab",
      'target="_blank"' in html)
check("Link has security attributes",
      'rel="noopener noreferrer"' in html)

# Section heading uses approved wording
check("Sources Analyzed heading present",
      "Sources Analyzed" in html or "Articles Compared" in html)

# Second article with missing optional fields still renders
check("Second article title displayed",
      "Turkey earthquake causes damage" in html)
check("Reuters source displayed", "Reuters" in html)
check("Second article URL is a link",
      'href="https://reuters.com/earthquake"' in html)


# ──────────────────────────────────────────────────────────────
# Test 11: Missing fields don't break rendering
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 11: Missing Fields Don't Break Rendering")
print("=" * 70)

# Create result with article that has NO optional fields at all
minimal_result = make_success_result()
minimal_result["groups"] = [
    {
        "group_id": 1,
        "label": "Minimal Test",
        "article_count": 1,
        "articles": [
            {
                "title": "Minimal article title only",
                "source": {"name": "TestSource"},
                # No author, no publishedAt, no description, no url
            },
        ],
    },
]

with patch("app.run_pipeline", return_value=minimal_result):
    response = client.post("/", data={"topic": "test"})
    html = response.data.decode("utf-8")

check("Page renders without crash (status 200)", response.status_code == 200)
check("Minimal article title displayed",
      "Minimal article title only" in html)
check("Minimal source displayed", "TestSource" in html)
check("No crash from missing author", 'id="result-output"' in html)
check("No crash from missing url", response.status_code == 200)


# ──────────────────────────────────────────────────────────────
# Test 12: Relevance Statistics Bar & Notice
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 12: Relevance Statistics Bar & Notice")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_success_result()):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("Articles Retrieved displayed", "Articles Retrieved" in html)
check("Filtered Out count displayed", "Filtered Out" in html)
check("Relevant Articles count displayed", "Relevant Articles" in html)
check("Relevance flow bar present", 'id="relevance-flow-bar"' in html)
check("Relevance notice present when filtered_out > 0",
      'id="relevance-notice"' in html and "Strict relevance filtering removed" in html)


# ──────────────────────────────────────────────────────────────
# Test 13: Zero Relevant Articles UI Error State
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 13: Zero Relevant Articles UI Error State")
print("=" * 70)

with patch("app.run_pipeline", return_value=make_zero_relevant_result()):
    response = client.post("/", data={"topic": "unrelatedtopic"})
    html = response.data.decode("utf-8")

check("Status code 200", response.status_code == 200)
check("Zero-relevant state box rendered", 'id="zero-relevant-state"' in html)
check("Title indicates no relevant articles found",
      "No relevant articles found" in html)
check("Subtitle mentions fetched count (8)", "8" in html)
check("Diagnostic note present", "Strict relevance filtering removed 8" in html)
check("Retry / new search action present", 'id="retry-search-btn"' in html)
check("No analysis output container rendered", 'id="result-output"' not in html)


# ──────────────────────────────────────────────────────────────
# Test 14: Multiple Event Groups & Sources Rendering
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 14: Multiple Event Groups & Sources Rendering")
print("=" * 70)

multi_group_result = make_success_result()
multi_group_result["event_groups"] = 2
multi_group_result["groups"] = [
    {
        "group_id": 1,
        "label": "First Event Group",
        "article_count": 1,
        "articles": [
            {
                "title": "First Event Article",
                "source": {"name": "Source A"},
                "url": "https://example.com/a",
            }
        ],
    },
    {
        "group_id": 2,
        "label": "Second Event Group",
        "article_count": 1,
        "articles": [
            {
                "title": "Second Event Article",
                "source": {"name": "Source B"},
                "url": "https://example.com/b",
            }
        ],
    },
]

with patch("app.run_pipeline", return_value=multi_group_result):
    response = client.post("/", data={"topic": "earthquake"})
    html = response.data.decode("utf-8")

check("First event label rendered", "First Event Group" in html)
check("First event article title rendered", "First Event Article" in html)
check("Second event label rendered", "Second Event Group" in html)
check("Second event article title rendered", "Second Event Article" in html)


# ──────────────────────────────────────────────────────────────
# Test 15: Missing URL Does Not Render Fake Link
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 15: Missing URL Does Not Render Fake Link")
print("=" * 70)

no_url_result = make_success_result()
no_url_result["groups"] = [
    {
        "group_id": 1,
        "label": "No URL Event",
        "article_count": 1,
        "articles": [
            {
                "title": "Article Without URL",
                "source": {"name": "Offline Source"},
                "url": None,  # No URL
            }
        ],
    }
]

with patch("app.run_pipeline", return_value=no_url_result):
    response = client.post("/", data={"topic": "test"})
    html = response.data.decode("utf-8")

check("Article title rendered", "Article Without URL" in html)
check("No href with fake URL rendered", 'href="None"' not in html and 'href=""' not in html)
check("No Read Original Article button for missing URL",
      "Read Original Article" not in html)


# ──────────────────────────────────────────────────────────────
# Test 16: Security Test - Secret Key Exclusion
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 16: Security Test - Secret Key Exclusion")
print("=" * 70)

response_get = client.get("/")
html_get = response_get.data.decode("utf-8")

check("NEWS_API_KEY absent from GET HTML", "NEWS_API_KEY" not in html_get)
check("GEMINI_API_KEY absent from GET HTML", "GEMINI_API_KEY" not in html_get)

with patch("app.run_pipeline", return_value=make_success_result()):
    response_post = client.post("/", data={"topic": "earthquake"})
    html_post = response_post.data.decode("utf-8")
    check("NEWS_API_KEY absent from POST HTML", "NEWS_API_KEY" not in html_post)
    check("GEMINI_API_KEY absent from POST HTML", "GEMINI_API_KEY" not in html_post)
    check(".env file references absent from HTML", ".env" not in html_post)


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")


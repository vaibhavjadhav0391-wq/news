"""
Test suite for Stage 1B: Strict Relevance Filter

All tests verify deterministic precision-first filtering behavior without external API calls.
"""

import sys
from processing.relevance_filter import (
    normalize_text,
    extract_query_concepts,
    calculate_relevance_score,
    filter_relevant_articles,
    DEFAULT_RELEVANCE_THRESHOLD,
)

passed = 0
failed = 0


def check(label, condition):
    """Simple test assertion helper."""
    global passed, failed
    if condition:
        print(f"  PASS: {label}")
        passed += 1
    else:
        print(f"  FAIL: {label}")
        failed += 1


# ──────────────────────────────────────────────────────────────
# Test 1: Text Normalization
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Text Normalization")
print("=" * 70)

check("Converts to lowercase", normalize_text("India AI") == "india ai")
check("Strips punctuation", normalize_text("India's AI regulations!") == "indias ai regulations")
check("Normalizes hyphens", normalize_text("multi-source news") == "multi source news")
check("Collapses whitespace", normalize_text("  India   AI   ") == "india ai")
check("Handles None gracefully", normalize_text(None) == "")
check("Handles empty string", normalize_text("") == "")


# ──────────────────────────────────────────────────────────────
# Test 2: Query Concept Extraction
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: Query Concept Extraction")
print("=" * 70)

res = extract_query_concepts("India AI regulations")
check("Tokens extracted", "india" in res["tokens"] and "ai" in res["tokens"])
check("Entities detected", len(res["anchor_entities"]) >= 1)
check("Entity is India", any(e["term"] == "india" for e in res["anchor_entities"]))
check("Query phrases generated", "india ai" in res["query_phrases"])

empty_res = extract_query_concepts("")
check("Empty query handles safely", empty_res["tokens"] == [])


# ──────────────────────────────────────────────────────────────
# Test 3: Core Query "India AI regulations" - KEEP cases
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 3: Core Query 'India AI regulations' - KEEP cases")
print("=" * 70)

query = "India AI regulations"

keep_1 = {
    "title": "India proposes new AI regulations",
    "description": "The Indian government released a draft framework for artificial intelligence rules.",
}
score_1 = calculate_relevance_score(query, keep_1)
check("Exact title match -> KEEP", score_1["passed"])
check("High score for exact match", score_1["score"] >= 0.70)

keep_2 = {
    "title": "Indian government considers rules for artificial intelligence",
    "description": "New Delhi plans legislation governing machine learning models.",
}
score_2 = calculate_relevance_score(query, keep_2)
check("Paraphrased match with aliases -> KEEP", score_2["passed"])
check("Score above threshold", score_2["score"] >= 0.50)

keep_3 = {
    "title": "India's AI policy framework takes shape",
    "description": "Government officials discuss regulations for synthetic intelligence.",
}
score_3 = calculate_relevance_score(query, keep_3)
check("Policy framework match -> KEEP", score_3["passed"])


# ──────────────────────────────────────────────────────────────
# Test 4: Core Query "India AI regulations" - REJECT cases
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 4: Core Query 'India AI regulations' - REJECT cases")
print("=" * 70)

reject_1 = {
    "title": "EU introduces new AI regulations",
    "description": "European Parliament passes landmark Artificial Intelligence Act.",
}
score_r1 = calculate_relevance_score(query, reject_1)
check("Wrong country entity (EU) -> REJECT", not score_r1["passed"])
check("Hard reject for missing India entity", "Missing critical entity" in str(score_r1["rejection_reason"]))

reject_2 = {
    "title": "India climate regulations updated",
    "description": "New environmental rules for manufacturing in India.",
}
score_r2 = calculate_relevance_score(query, reject_2)
check("Entity matches but zero AI match -> REJECT", not score_r2["passed"])

reject_3 = {
    "title": "AI weather prediction in Europe",
    "description": "European scientists use machine learning models for weather forecasting.",
}
score_r3 = calculate_relevance_score(query, reject_3)
check("AI in Europe -> REJECT", not score_r3["passed"])

reject_4 = {
    "title": "Weekly Climate and Energy News Roundup",
    "description": "A summary of energy policies in Asia and Europe.",
}
score_r4 = calculate_relevance_score(query, reject_4)
check("Completely unrelated roundup -> REJECT", not score_r4["passed"])


# ──────────────────────────────────────────────────────────────
# Test 5: OpenAI Entity Hard-Reject Rule
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Entity Hard-Reject (OpenAI vs Google)")
print("=" * 70)

q_openai = "OpenAI latest model"

keep_openai = {
    "title": "OpenAI announces its latest AI model",
    "description": "OpenAI introduced GPT 5 with improved reasoning capabilities.",
}
score_o1 = calculate_relevance_score(q_openai, keep_openai)
check("OpenAI article -> KEEP", score_o1["passed"])

reject_openai = {
    "title": "Google announces a new AI model",
    "description": "Alphabet unveiled Gemini Ultra for enterprise applications.",
}
score_o2 = calculate_relevance_score(q_openai, reject_openai)
check("Google model article for OpenAI query -> REJECT", not score_o2["passed"])
check("Hard reject reason mentions OpenAI", "OpenAI" in str(score_o2["rejection_reason"]) or not score_o2["passed"])


# ──────────────────────────────────────────────────────────────
# Test 6: Apple vs Samsung Entity Hard-Reject Rule
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 6: Entity Hard-Reject (Apple vs Samsung)")
print("=" * 70)

q_apple = "Apple iPhone launch"

keep_apple = {
    "title": "Apple announces new iPhone",
    "description": "Apple held its keynote event in Cupertino to launch the iPhone 17.",
}
score_a1 = calculate_relevance_score(q_apple, keep_apple)
check("Apple iPhone article -> KEEP", score_a1["passed"])

reject_apple = {
    "title": "Samsung launches new Galaxy phone",
    "description": "Samsung Electronics unveiled the Galaxy S26 with AI features.",
}
score_a2 = calculate_relevance_score(q_apple, reject_apple)
check("Samsung Galaxy article for Apple iPhone query -> REJECT", not score_a2["passed"])


# ──────────────────────────────────────────────────────────────
# Test 7: Title Match Priority vs Description Only Match
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Title Priority vs Description Only Match")
print("=" * 70)

query_p = "India AI regulations"

art_title_match = {
    "title": "India announces new AI regulations",
    "description": "A brief overview of technology news.",
}

art_desc_only = {
    "title": "Technology conference held in Delhi",
    "description": "Officials discussed AI regulations in India during the panel.",
}

score_title = calculate_relevance_score(query_p, art_title_match)
score_desc = calculate_relevance_score(query_p, art_desc_only)

check("Title match scores higher than description-only match",
      score_title["score"] > score_desc["score"])


# ──────────────────────────────────────────────────────────────
# Test 8: Robustness & Edge Cases
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Robustness & Edge Cases")
print("=" * 70)

# Empty query
res_empty_q = filter_relevant_articles("", [keep_1])
check("Empty query -> 0 accepted", res_empty_q["stats"]["accepted"] == 0)

# Empty article list
res_empty_arts = filter_relevant_articles("India AI", [])
check("Empty article list -> 0 total", res_empty_arts["stats"]["total_articles"] == 0)

# Missing title
art_no_title = {
    "title": None,
    "description": "India proposes new AI regulations and policy rules.",
}
score_no_title = calculate_relevance_score("India AI", art_no_title)
check("Missing title handled safely without exception", True)

# Missing description
art_no_desc = {
    "title": "India proposes new AI regulations",
    "description": None,
}
score_no_desc = calculate_relevance_score("India AI regulations", art_no_desc)
check("Missing description handled safely", score_no_desc["passed"])

# Missing both
art_missing_both = {"title": None, "description": None}
score_both_missing = calculate_relevance_score("India AI", art_missing_both)
check("Missing both -> REJECT", not score_both_missing["passed"])

# Missing URL
art_no_url = {
    "title": "India proposes new AI regulations",
    "description": "Official rules released.",
    "source": {"name": "BBC"},
}
res_no_url = filter_relevant_articles("India AI", [art_no_url])
check("Missing URL handled without error", res_no_url["stats"]["accepted"] == 1)

# Malformed article object
res_malformed = filter_relevant_articles("India AI", [None, "not a dict", {}])
check("Malformed items handled safely", res_malformed["stats"]["accepted"] == 0)


# ──────────────────────────────────────────────────────────────
# Test 9: Batch Filtering Statistics & Original Article Preserved
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 9: Batch Filtering & Article Preservation")
print("=" * 70)

batch_input = [keep_1, keep_2, reject_1, reject_2, reject_3, reject_4]
batch_res = filter_relevant_articles("India AI regulations", batch_input)

check("Total articles counted correctly", batch_res["stats"]["total_articles"] == 6)
check("Accepted count is 2", batch_res["stats"]["accepted"] == 2)
check("Rejected count is 4", batch_res["stats"]["rejected"] == 4)

accepted = batch_res["relevant_articles"]
check("Accepted article 1 is original object", accepted[0]["title"] == keep_1["title"])
check("Accepted article 2 is original object", accepted[1]["title"] == keep_2["title"])

rejected = batch_res["rejected_articles"]
check("Rejected articles contain relevance_score metadata", "relevance_score" in rejected[0])
check("Rejected articles contain rejection_reason metadata", "rejection_reason" in rejected[0])


# ──────────────────────────────────────────────────────────────
# Test 10: Threshold Tuning & Borderline Cases
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 10: Threshold Tuning & Borderline Cases")
print("=" * 70)

# Article that is vaguely related to technology in India, but not AI regulations
borderline = {
    "title": "India tech sector grows rapidly",
    "description": "Software companies in Bangalore see increased investment.",
}
score_borderline = calculate_relevance_score("India AI regulations", borderline)
check("Borderline tech article rejected under default threshold", not score_borderline["passed"])


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")

if failed > 0:
    sys.exit(1)

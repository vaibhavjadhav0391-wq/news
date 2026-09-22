"""
Test script for the end-to-end pipeline orchestrator.

All tests use MOCKED NewsAPI and Gemini API responses -- no real API calls.
"""

import json
from unittest.mock import MagicMock, patch

from pipeline import run_pipeline

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


def make_articles(n=3, topic="earthquake"):
    """Generate n fake articles about the same topic."""
    articles = []
    sources = ["BBC", "Reuters", "Al Jazeera", "NDTV", "AP"]
    for i in range(n):
        source = sources[i % len(sources)]
        articles.append({
            "title": f"{topic.title()} report from {source} - article {i+1}",
            "description": (
                f"Details about the {topic} event reported by {source}. "
                f"This covers the latest updates on the {topic}."
            ),
            "source": {"name": source},
            "author": f"Author {i+1}",
            "url": f"https://example.com/{i+1}",
            "publishedAt": "2026-08-23T10:00:00Z",
        })
    return articles


def make_duplicate_articles():
    """Generate articles that are exact/near duplicates."""
    return [
        {
            "title": "Earthquake strikes eastern Turkey",
            "description": "A powerful earthquake struck eastern Turkey today causing damage.",
            "source": {"name": "BBC"},
            "url": "https://example.com/1",
        },
        {
            "title": "Earthquake strikes eastern Turkey",
            "description": "A powerful earthquake struck eastern Turkey today causing damage.",
            "source": {"name": "Mirror"},
            "url": "https://example.com/2",
        },
    ]


def make_multi_event_articles():
    """Generate articles about two distinct events."""
    return [
        {
            "title": "India launches Chandrayaan-5 lunar mission",
            "description": (
                "ISRO successfully launched Chandrayaan-5 from Sriharikota "
                "for Moon exploration."
            ),
            "source": {"name": "Times of India"},
        },
        {
            "title": "ISRO Chandrayaan-5 mission lifts off to the Moon",
            "description": (
                "India's Chandrayaan-5 spacecraft launched from Sriharikota "
                "targeting lunar south pole."
            ),
            "source": {"name": "NDTV"},
        },
        {
            "title": "Apple unveils new MacBook with M5 processor",
            "description": (
                "Apple announced the next generation MacBook featuring "
                "the M5 chip with improved performance."
            ),
            "source": {"name": "TechCrunch"},
        },
        {
            "title": "New Apple MacBook M5 launched at keynote event",
            "description": (
                "Apple launched the MacBook M5 at its annual keynote, "
                "featuring a new M5 processor and better battery."
            ),
            "source": {"name": "The Verge"},
        },
    ]


def mock_fetch(articles):
    """Return a fetch function that returns the given articles."""
    def _fetch(topic, page_size=10):
        return articles
    return _fetch


def mock_gemini_response():
    """Return a valid mock Gemini JSON response text."""
    return json.dumps({
        "summary": "A test event occurred according to multiple sources.",
        "agreed_information": ["The event occurred"],
        "differing_information": [],
        "potential_contradictions": [],
        "unique_claims": [],
        "source_assessment": [
            {"source": "BBC", "assessment": "Detailed coverage"},
        ],
        "uncertainty_notes": ["Exact details unconfirmed"],
    })


def make_mock_gemini_client():
    """Create a mock Gemini client that returns valid JSON."""
    client = MagicMock()
    response = MagicMock()
    response.text = mock_gemini_response()
    client.models.generate_content.return_value = response
    return client


# ──────────────────────────────────────────────────────────────
# Test 1: Successful complete pipeline
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Successful Complete Pipeline")
print("=" * 70)

articles = make_articles(3)
mock_client = make_mock_gemini_client()

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(articles),
        gemini_client=mock_client,
    )

check("Status is success", result["status"] == "success")
check("No error", result["error"] is None)
check("Failed stage is None", result["failed_stage"] is None)
check("Topic recorded", result["topic"] == "earthquake")
check("Articles fetched = 3", result["articles_fetched"] == 3)
check("Unique articles > 0", result["unique_articles"] > 0)
check("Event groups > 0", result["event_groups"] > 0)
check("Comparisons produced", len(result["comparisons"]) > 0)
check("Analyses produced", len(result["analyses"]) > 0)
check("Gemini was called", mock_client.models.generate_content.call_count > 0)


# ──────────────────────────────────────────────────────────────
# Test 2: Empty input
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: Empty Input")
print("=" * 70)

result = run_pipeline("")
check("Status is error", result["status"] == "error")
check("Failed at input stage", result["failed_stage"] == "input")
check("Error mentions topic", "topic" in result["error"].lower())

result2 = run_pipeline(None)
check("None input -> error", result2["status"] == "error")

result3 = run_pipeline("   ")
check("Whitespace input -> error", result3["status"] == "error")


# ──────────────────────────────────────────────────────────────
# Test 3: No articles returned
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 3: No Articles Found")
print("=" * 70)

result = run_pipeline(
    "xyznonexistenttopic123",
    _fetch_fn=mock_fetch([]),
)

check("Status is error", result["status"] == "error")
check("Failed at fetch stage", result["failed_stage"] == "fetch")
check("Error mentions no articles", "no articles" in result["error"].lower())
check("Articles fetched = 0", result["articles_fetched"] == 0)


# ──────────────────────────────────────────────────────────────
# Test 4: Duplicate articles
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 4: Duplicate Articles")
print("=" * 70)

dupes = make_duplicate_articles()

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(dupes),
        gemini_client=make_mock_gemini_client(),
    )

check("Pipeline handled duplicates",
      result["status"] in ("success", "partial", "error"))
check("Articles fetched = 2", result["articles_fetched"] == 2)
check("Duplicates removed >= 0", result["duplicates_removed"] >= 0)

if result["status"] in ("success", "partial"):
    # These two articles are exact/near-duplicate text about the same event,
    # so exactly one of them must be reported as a removed duplicate. This
    # guards against duplicates_removed being read from the wrong key in
    # dedup_result (it silently defaulted to 0 regardless of actual dedup
    # behavior until this was fixed).
    check("Duplicates removed == 1 for exact-duplicate articles",
          result["duplicates_removed"] == 1)
    check("unique_articles + duplicates_removed == articles_after_relevance_filter",
          result["unique_articles"] + result["duplicates_removed"]
          == result["articles_after_relevance_filter"])

if result["status"] == "success":
    check("At least 1 unique article remained",
          result["unique_articles"] >= 1)


# ──────────────────────────────────────────────────────────────
# Test 5: Multiple event groups
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Multiple Event Groups")
print("=" * 70)

multi_articles = make_multi_event_articles()
mock_client = make_mock_gemini_client()

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "technology",
        _fetch_fn=mock_fetch(multi_articles),
        gemini_client=mock_client,
    )

check("Status is success", result["status"] == "success")
check("Articles fetched = 4", result["articles_fetched"] == 4)
check("Multiple event groups formed", result["event_groups"] >= 2)
check("Comparisons match groups",
      len(result["comparisons"]) == result["event_groups"])
check("Analyses match groups",
      len(result["analyses"]) == result["event_groups"])
check("Gemini called per group",
      mock_client.models.generate_content.call_count == result["event_groups"])


# ──────────────────────────────────────────────────────────────
# Test 6: Fetch failure (API error)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 6: Fetch Failure")
print("=" * 70)


def failing_fetch(topic, page_size=10):
    raise ConnectionError("Network unreachable")


result = run_pipeline("earthquake", _fetch_fn=failing_fetch)
check("Status is error", result["status"] == "error")
check("Failed at fetch stage", result["failed_stage"] == "fetch")
check("Error message present",
      "network" in result["error"].lower()
      or "fetch" in result["error"].lower())


# ──────────────────────────────────────────────────────────────
# Test 7: Gemini failure (partial success)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Gemini Failure (Partial Success)")
print("=" * 70)

failing_client = MagicMock()
failing_client.models.generate_content.side_effect = Exception(
    "Gemini API down"
)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(make_articles(3)),
        gemini_client=failing_client,
    )

check("Status is partial or error",
      result["status"] in ("partial", "error"))
check("Comparisons still produced", len(result["comparisons"]) > 0)
check("Error mentions Gemini", "gemini" in result["error"].lower())
check("Articles fetched recorded", result["articles_fetched"] == 3)
check("Event groups recorded", result["event_groups"] > 0)


# ──────────────────────────────────────────────────────────────
# Test 8: Malformed article data
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Malformed Article Data")
print("=" * 70)

malformed = [
    {"title": None, "description": None, "source": {"name": "A"}},
    {"title": "Real article", "description": "With content",
     "source": {"name": "B"}},
    {},  # completely empty dict
]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    try:
        result = run_pipeline(
            "test",
            _fetch_fn=mock_fetch(malformed),
            gemini_client=make_mock_gemini_client(),
        )
        check("Handles malformed data without crash", True)
        check("Status is success or error",
              result["status"] in ("success", "partial", "error"))
        check("Articles fetched = 3", result["articles_fetched"] == 3)
    except Exception as e:
        check(f"Crashed on malformed data: {e}", False)


# ──────────────────────────────────────────────────────────────
# Test 9: Skip Gemini mode
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 9: Skip Gemini Mode")
print("=" * 70)

result = run_pipeline(
    "earthquake",
    _fetch_fn=mock_fetch(make_articles(3)),
    skip_gemini=True,
)

check("Status is success", result["status"] == "success")
check("No analyses (Gemini skipped)", len(result["analyses"]) == 0)
check("Comparisons still produced", len(result["comparisons"]) > 0)
check("Event groups formed", result["event_groups"] > 0)


# ──────────────────────────────────────────────────────────────
# Test 10: Output structure validation
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 10: Output Structure Validation")
print("=" * 70)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(make_articles(3)),
        gemini_client=make_mock_gemini_client(),
    )

expected_keys = [
    "status", "failed_stage", "topic", "articles_fetched",
    "duplicates_removed", "unique_articles", "event_groups",
    "groups", "comparisons", "analyses", "error",
]

for key in expected_keys:
    check(f"Has key '{key}'", key in result)

check("status is str", isinstance(result["status"], str))
check("topic is str", isinstance(result["topic"], str))
check("articles_fetched is int",
      isinstance(result["articles_fetched"], int))
check("duplicates_removed is int",
      isinstance(result["duplicates_removed"], int))
check("unique_articles is int",
      isinstance(result["unique_articles"], int))
check("event_groups is int",
      isinstance(result["event_groups"], int))
check("comparisons is list",
      isinstance(result["comparisons"], list))
check("analyses is list",
      isinstance(result["analyses"], list))


# ──────────────────────────────────────────────────────────────
# Test 11: Single article pipeline
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 11: Single Article Pipeline")
print("=" * 70)

single = [make_articles(1)[0]]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(single),
        gemini_client=make_mock_gemini_client(),
    )

check("Status is success", result["status"] == "success")
check("Articles fetched = 1", result["articles_fetched"] == 1)
check("Unique articles = 1", result["unique_articles"] == 1)
check("1 event group", result["event_groups"] == 1)
check("1 comparison", len(result["comparisons"]) == 1)


# ──────────────────────────────────────────────────────────────
# Test 12: Missing Gemini API key
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 12: Missing Gemini API Key")
print("=" * 70)

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", None):
    result = run_pipeline(
        "earthquake",
        _fetch_fn=mock_fetch(make_articles(3)),
    )

check("Status is partial (stages 1-4 ok, Gemini failed)",
      result["status"] == "partial")
check("Comparisons still available",
      len(result["comparisons"]) > 0)
check("Error mentions API key or Gemini",
      "api key" in result["error"].lower()
      or "gemini" in result["error"].lower())


# ──────────────────────────────────────────────────────────────
# Test 13: Groups contain article data (for frontend rendering)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 13: Groups Contain Article Data")
print("=" * 70)

# Use articles with ALL fields populated
full_articles = [
    {
        "title": "Climate summit opens in Berlin",
        "description": "World leaders gather for climate talks in Berlin.",
        "source": {"name": "BBC News"},
        "author": "Jane Reporter",
        "url": "https://bbc.com/news/climate-summit",
        "publishedAt": "2026-09-01T08:00:00Z",
    },
    {
        "title": "Berlin climate summit begins with key speeches",
        "description": "Climate summit in Berlin opens with speeches from leaders.",
        "source": {"name": "Reuters"},
        "author": "John Journalist",
        "url": "https://reuters.com/climate-summit",
        "publishedAt": "2026-09-01T09:30:00Z",
    },
]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    result = run_pipeline(
        "climate",
        _fetch_fn=mock_fetch(full_articles),
        gemini_client=make_mock_gemini_client(),
    )

check("Result has groups field", "groups" in result)
check("Groups is a list", isinstance(result["groups"], list))
check("Groups is not empty", len(result["groups"]) > 0)

# Check first group structure
group = result["groups"][0]
check("Group has group_id", "group_id" in group)
check("Group has label", "label" in group)
check("Group has articles", "articles" in group)
check("Group has article_count", "article_count" in group)
check("Group articles is a list", isinstance(group["articles"], list))
check("Group articles not empty", len(group["articles"]) > 0)

# Check article fields are preserved
article = group["articles"][0]
check("Article has title", "title" in article)
check("Article title is a string", isinstance(article["title"], str))
check("Article title not empty", len(article["title"]) > 0)

check("Article has source", "source" in article)
check("Article source has name",
      isinstance(article["source"], dict) and "name" in article["source"])

check("Article has url", "url" in article)
check("Article url is a string", isinstance(article["url"], str))
check("Article url starts with http", article["url"].startswith("http"))

check("Article has author", "author" in article)
check("Article has publishedAt", "publishedAt" in article)
check("Article has description", "description" in article)

# Verify specific values are preserved (not mangled)
all_titles = [a["title"] for g in result["groups"] for a in g["articles"]]
check("Original title preserved",
      any("climate" in t.lower() or "Climate" in t for t in all_titles))

all_urls = [a.get("url", "") for g in result["groups"] for a in g["articles"]]
check("Original URL preserved",
      any("climate-summit" in u for u in all_urls))

all_authors = [a.get("author", "") for g in result["groups"]
               for a in g["articles"]]
check("Author preserved",
      any("Reporter" in str(a) or "Journalist" in str(a) for a in all_authors))

all_dates = [a.get("publishedAt", "") for g in result["groups"]
             for a in g["articles"]]
check("PublishedAt preserved",
      any("2026-09-01" in str(d) for d in all_dates))

# Verify groups count matches event_groups
check("Groups count matches event_groups",
      len(result["groups"]) == result["event_groups"])


# ──────────────────────────────────────────────────────────────
# Test 14: Groups with missing article fields (graceful handling)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 14: Groups With Missing Article Fields")
print("=" * 70)

partial_articles = [
    {
        "title": "Article with no author or date",
        "description": "Some description but no author or date fields.",
        "source": {"name": "TestSource"},
        "url": "https://example.com/no-author",
        # No author, no publishedAt
    },
    {
        "title": "Article with minimal fields only",
        "source": {"name": "MinimalSource"},
        # No description, no author, no publishedAt, no url
    },
]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    try:
        result = run_pipeline(
            "test",
            _fetch_fn=mock_fetch(partial_articles),
            gemini_client=make_mock_gemini_client(),
        )
        check("Handles missing fields without crash", True)
        check("Groups still present", "groups" in result)
        if result["groups"]:
            arts = result["groups"][0]["articles"]
            check("Articles still present", len(arts) > 0)
            # Check graceful absence
            art_no_author = next(
                (a for a in arts if a.get("title", "").startswith("Article with no")),
                None
            )
            if art_no_author:
                check("Missing author returns None or absent",
                      art_no_author.get("author") is None
                      or "author" not in art_no_author)
    except Exception as e:
        check(f"Crashed on partial data: {e}", False)


# ──────────────────────────────────────────────────────────────
# Test 15: Error results have empty groups
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 15: Error Results Have Empty Groups")
print("=" * 70)

result = run_pipeline("", _fetch_fn=mock_fetch([]))
check("Error result has groups field", "groups" in result)
check("Error result groups is empty list", result["groups"] == [])

result2 = run_pipeline("nothing", _fetch_fn=mock_fetch([]))
check("No-articles result has empty groups", result2["groups"] == [])


# ──────────────────────────────────────────────────────────────
# Test 16: Relevance Filter Pipeline Integration (Stage 1B)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 16: Relevance Filter Pipeline Integration (Stage 1B)")
print("=" * 70)

mixed_articles = [
    {
        "title": "India proposes new AI regulations",
        "description": "Indian government released a draft framework for artificial intelligence.",
        "source": {"name": "BBC"},
        "url": "https://example.com/india-ai-1",
    },
    {
        "title": "EU introduces new AI regulations",
        "description": "European Parliament passes landmark Artificial Intelligence Act.",
        "source": {"name": "Reuters"},
        "url": "https://example.com/eu-ai-1",
    },
    {
        "title": "Weekly Climate and Energy News Roundup",
        "description": "A summary of energy policies in Asia and Europe.",
        "source": {"name": "Climate Times"},
        "url": "https://example.com/climate-1",
    },
]

with patch("analysis.gemini_analyzer.GEMINI_API_KEY", "test_key"):
    res_filtered = run_pipeline(
        "India AI regulations",
        _fetch_fn=mock_fetch(mixed_articles),
        gemini_client=make_mock_gemini_client(),
    )

check("Pipeline contains articles_filtered_out key", "articles_filtered_out" in res_filtered)
check("Pipeline contains articles_after_relevance_filter key", "articles_after_relevance_filter" in res_filtered)
check("Articles fetched = 3", res_filtered["articles_fetched"] == 3)
check("Articles filtered out = 2", res_filtered["articles_filtered_out"] == 2)
check("Articles after filter = 1", res_filtered["articles_after_relevance_filter"] == 1)

# Test zero relevant articles handling
only_irrelevant = [
    {
        "title": "EU introduces new AI regulations",
        "description": "European Parliament passes landmark Artificial Intelligence Act.",
        "source": {"name": "Reuters"},
    },
]

res_zero_rel = run_pipeline(
    "India AI regulations",
    _fetch_fn=mock_fetch(only_irrelevant),
    gemini_client=make_mock_gemini_client(),
)

check("Zero relevant articles status is error", res_zero_rel["status"] == "error")
check("Failed stage is relevance_filter", res_zero_rel["failed_stage"] == "relevance_filter")
check("Error message mentions relevance", "relevant" in res_zero_rel["error"].lower())


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")


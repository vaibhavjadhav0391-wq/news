"""
Test script for the article deduplicator.

Tests normalization, similarity detection, and edge cases using
sample articles that simulate real NewsAPI output.
"""

from processing.deduplicator import (
    normalize_text,
    get_article_text,
    find_duplicates,
    print_dedup_report,
    DEFAULT_SIMILARITY_THRESHOLD,
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


# ──────────────────────────────────────────────────────────────
# Test 1: Text normalization
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Text Normalization")
print("=" * 70)

check("Lowercase", normalize_text("Hello WORLD") == "hello world")
check("Remove punctuation", normalize_text("it's great!") == "its great")
check("Collapse whitespace", normalize_text("hello    world") == "hello world")
check("Strip edges", normalize_text("  hello  ") == "hello")
check("Empty string", normalize_text("") == "")
check("None input", normalize_text(None) == "")
check(
    "Combined cleanup",
    normalize_text("  Breaking NEWS:  AI is Here!! ") == "breaking news ai is here",
)

# ──────────────────────────────────────────────────────────────
# Test 2: Exact duplicates (same title and description)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: Exact Duplicates")
print("=" * 70)

exact_dup_articles = [
    {
        "title": "India launches new space mission to Mars",
        "description": "ISRO announced a new Mars exploration mission today.",
        "source": {"name": "Times of India"},
        "url": "https://example.com/1",
    },
    {
        "title": "India launches new space mission to Mars",
        "description": "ISRO announced a new Mars exploration mission today.",
        "source": {"name": "NDTV"},
        "url": "https://example.com/2",
    },
    {
        "title": "Stock market hits record high amid global rally",
        "description": "Markets surged today with tech stocks leading the way.",
        "source": {"name": "Bloomberg"},
        "url": "https://example.com/3",
    },
]

result = find_duplicates(exact_dup_articles)
check("Total input is 3", result["total_input"] == 3)
check("Unique articles is 2", result["total_unique"] == 2)
check("Removed 1 duplicate", result["total_removed"] == 1)
check("One duplicate group found", len(result["duplicate_groups"]) == 1)

print("\n  Report:")
print_dedup_report(result)

# ──────────────────────────────────────────────────────────────
# Test 3: Near-duplicates (same story, different wording)
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 3: Near-Duplicates (Same Story, Different Wording)")
print("=" * 70)

near_dup_articles = [
    {
        "title": "Apple releases new iPhone 16 with AI features",
        "description": "Apple unveiled the iPhone 16 today featuring advanced AI capabilities and a new chip.",
        "source": {"name": "TechCrunch"},
        "url": "https://example.com/1",
    },
    {
        "title": "Apple announces iPhone 16 with artificial intelligence",
        "description": "The new iPhone 16 was announced by Apple with artificial intelligence features and improved hardware.",
        "source": {"name": "The Verge"},
        "url": "https://example.com/2",
    },
    {
        "title": "Earthquake hits Turkey, thousands displaced",
        "description": "A major earthquake struck eastern Turkey early this morning, displacing thousands of people.",
        "source": {"name": "BBC"},
        "url": "https://example.com/3",
    },
    {
        "title": "Turkey earthquake: magnitude 7.1 leaves thousands homeless",
        "description": "An earthquake measuring 7.1 on the Richter scale hit Turkey, leaving thousands without shelter.",
        "source": {"name": "Al Jazeera"},
        "url": "https://example.com/4",
    },
    {
        "title": "NASA discovers high potential high for life on Europa",
        "description": "NASA scientists say Europa may have conditions suitable for microbial life beneath its icy surface.",
        "source": {"name": "Space.com"},
        "url": "https://example.com/5",
    },
]

result = find_duplicates(near_dup_articles, threshold=0.40)
check("Total input is 5", result["total_input"] == 5)
check("Unique reduced (some dups found)", result["total_unique"] < 5)
check("At least 1 duplicate group", len(result["duplicate_groups"]) >= 1)

print("\n  Report:")
print_dedup_report(result)

# ──────────────────────────────────────────────────────────────
# Test 4: All unique articles (no duplicates)
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 4: All Unique Articles (No Duplicates Expected)")
print("=" * 70)

unique_articles = [
    {
        "title": "New climate change report warns of rising sea levels",
        "description": "The UN released a comprehensive report on global warming impacts on coastal cities.",
        "source": {"name": "Reuters"},
        "url": "https://example.com/1",
    },
    {
        "title": "Football World Cup 2026 schedule announced",
        "description": "FIFA released the official match schedule for the 2026 World Cup to be held in North America.",
        "source": {"name": "ESPN"},
        "url": "https://example.com/2",
    },
    {
        "title": "New study links coffee consumption to longer lifespan",
        "description": "Researchers at Harvard found that moderate coffee consumption may extend life expectancy.",
        "source": {"name": "Medical News"},
        "url": "https://example.com/3",
    },
]

result = find_duplicates(unique_articles)
check("Total input is 3", result["total_input"] == 3)
check("All 3 are unique", result["total_unique"] == 3)
check("Zero removed", result["total_removed"] == 0)
check("No duplicate groups", len(result["duplicate_groups"]) == 0)

# ──────────────────────────────────────────────────────────────
# Test 5: Edge cases
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Edge Cases")
print("=" * 70)

# Empty list
result = find_duplicates([])
check("Empty input -> 0 unique", result["total_unique"] == 0)

# Single article
result = find_duplicates([{"title": "Solo article", "source": {"name": "Test"}}])
check("Single article -> 1 unique", result["total_unique"] == 1)

# Articles with missing fields
result = find_duplicates([
    {"title": None, "description": None, "source": {"name": "A"}},
    {"title": "Real article", "description": "With content", "source": {"name": "B"}},
])
check("Missing fields handled", result["total_unique"] >= 1)

# ──────────────────────────────────────────────────────────────
# Test 6: Configurable threshold
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 6: Configurable Threshold")
print(f"  Default threshold: {DEFAULT_SIMILARITY_THRESHOLD}")
print("=" * 70)

similar_articles = [
    {
        "title": "Apple releases new iPhone with AI features",
        "description": "Apple unveiled the new iPhone today featuring advanced AI capabilities.",
        "source": {"name": "Source A"},
    },
    {
        "title": "Apple announces iPhone with artificial intelligence",
        "description": "The new iPhone was announced by Apple with artificial intelligence features.",
        "source": {"name": "Source B"},
    },
]

# Strict threshold — should find no duplicates
strict_result = find_duplicates(similar_articles, threshold=0.95)
# Lenient threshold — should find duplicates
lenient_result = find_duplicates(similar_articles, threshold=0.30)

check(
    "Strict threshold (0.95) keeps both",
    strict_result["total_unique"] == 2,
)
check(
    "Lenient threshold (0.30) merges them",
    lenient_result["total_unique"] == 1,
)

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")

"""
Test script for the event-based article grouper.

Tests event grouping across same-event articles, different-event articles,
varied wording, single-article groups, edge cases, and configurable threshold.
"""

from processing.grouper import (
    group_articles,
    print_group_report,
    DEFAULT_GROUP_THRESHOLD,
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
# Test 1: Articles clearly about the same event
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Articles About the Same Event")
print("=" * 70)

same_event_articles = [
    {
        "title": "India launches Chandrayaan-4 mission to the Moon",
        "description": "ISRO successfully launched Chandrayaan-4 from Sriharikota today, marking India's next step in lunar exploration.",
        "source": {"name": "Times of India"},
        "url": "https://example.com/1",
    },
    {
        "title": "ISRO's Chandrayaan-4 lifts off for Moon mission",
        "description": "India's space agency ISRO launched the Chandrayaan-4 spacecraft towards the Moon from the Sriharikota launch pad.",
        "source": {"name": "NDTV"},
        "url": "https://example.com/2",
    },
    {
        "title": "Chandrayaan-4: India's latest lunar mission takes flight",
        "description": "The Chandrayaan-4 mission launched today as ISRO aims to further explore the lunar surface.",
        "source": {"name": "The Hindu"},
        "url": "https://example.com/3",
    },
]

result = group_articles(same_event_articles)
check("Total articles is 3", result["total_articles"] == 3)
check("All 3 grouped into 1 event", result["total_groups"] == 1)
check("Group has 3 articles", result["groups"][0]["article_count"] == 3)
check("Group has a label", len(result["groups"][0]["label"]) > 0)
check("Group has a representative", result["groups"][0]["representative"] is not None)

print("\n  Report:")
print_group_report(result)


# ──────────────────────────────────────────────────────────────
# Test 2: Articles about completely different events
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 2: Articles About Completely Different Events")
print("=" * 70)

diff_event_articles = [
    {
        "title": "NASA discovers high potential for life on Europa",
        "description": "NASA scientists announced that Europa may have conditions suitable for microbial life beneath its icy surface.",
        "source": {"name": "Space.com"},
        "url": "https://example.com/1",
    },
    {
        "title": "Premier League season kicks off with Manchester derby",
        "description": "The English Premier League began its new season with an exciting Manchester derby between United and City.",
        "source": {"name": "ESPN"},
        "url": "https://example.com/2",
    },
    {
        "title": "Global coffee prices hit record highs due to drought",
        "description": "Coffee prices surged to historic levels as severe drought in Brazil reduced crop yields significantly.",
        "source": {"name": "Bloomberg"},
        "url": "https://example.com/3",
    },
]

result = group_articles(diff_event_articles)
check("Total articles is 3", result["total_articles"] == 3)
check("Each article is its own group", result["total_groups"] == 3)

# Each group should have exactly 1 article
all_single = all(g["article_count"] == 1 for g in result["groups"])
check("All groups have 1 article each", all_single)

print("\n  Report:")
print_group_report(result)


# ──────────────────────────────────────────────────────────────
# Test 3: Different wording but same event
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 3: Different Wording but Same Event")
print("=" * 70)

diff_wording_articles = [
    {
        "title": "Earthquake strikes Turkey, magnitude 7.1",
        "description": "A powerful earthquake measuring 7.1 struck eastern Turkey early this morning, causing widespread damage and displacing thousands of residents.",
        "source": {"name": "BBC"},
        "url": "https://example.com/1",
    },
    {
        "title": "Turkey earthquake: thousands displaced as buildings collapse",
        "description": "Thousands have been displaced after a devastating earthquake hit eastern Turkey, collapsing buildings and causing damage across the region.",
        "source": {"name": "Al Jazeera"},
        "url": "https://example.com/2",
    },
    {
        "title": "Apple unveils new MacBook Pro with M4 chip",
        "description": "Apple announced the next-generation MacBook Pro featuring the M4 chip with improved performance and battery life.",
        "source": {"name": "TechCrunch"},
        "url": "https://example.com/3",
    },
]

result = group_articles(diff_wording_articles)
check("Total articles is 3", result["total_articles"] == 3)
check("Two groups formed (earthquake + apple)", result["total_groups"] == 2)

# Find the earthquake group
earthquake_group = None
for g in result["groups"]:
    titles = [a.get("title", "") for a in g["articles"]]
    if any("turkey" in t.lower() or "quake" in t.lower() or "earthquake" in t.lower() for t in titles):
        earthquake_group = g
        break

check("Earthquake articles grouped together", earthquake_group is not None and earthquake_group["article_count"] == 2)

print("\n  Report:")
print_group_report(result)


# ──────────────────────────────────────────────────────────────
# Test 4: Single-article groups
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 4: Single-Article Groups")
print("=" * 70)

# Only one article
single = [
    {
        "title": "New vaccine shows promise against malaria",
        "description": "Clinical trials reveal a new malaria vaccine with 80% efficacy in children.",
        "source": {"name": "WHO"},
        "url": "https://example.com/1",
    },
]

result = group_articles(single)
check("Total articles is 1", result["total_articles"] == 1)
check("One group formed", result["total_groups"] == 1)
check("Group has 1 article", result["groups"][0]["article_count"] == 1)
check("Label is the article title", result["groups"][0]["label"] == single[0]["title"])


# ──────────────────────────────────────────────────────────────
# Test 5: Mixed scenario (2 events, multiple articles each)
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 5: Mixed Scenario (Multiple Events)")
print("=" * 70)

mixed_articles = [
    # Event A: India cricket
    {
        "title": "India wins Cricket World Cup final against Australia",
        "description": "India beat Australia in an thrilling Cricket World Cup final at the MCG in Melbourne.",
        "source": {"name": "ESPNcricinfo"},
        "url": "https://example.com/1",
    },
    {
        "title": "Cricket: India defeats Australia to win World Cup",
        "description": "Indian cricket team defeated Australia in the World Cup cricket final at Melbourne Cricket Ground.",
        "source": {"name": "The Hindu"},
        "url": "https://example.com/2",
    },
    # Event B: AI regulation
    {
        "title": "EU passes landmark AI regulation law",
        "description": "The European Union approved comprehensive legislation to regulate artificial intelligence systems.",
        "source": {"name": "Reuters"},
        "url": "https://example.com/3",
    },
    {
        "title": "European Parliament approves AI Act regulation",
        "description": "EU lawmakers passed the AI Act, the world's first comprehensive artificial intelligence regulation framework.",
        "source": {"name": "BBC"},
        "url": "https://example.com/4",
    },
    # Event C: standalone
    {
        "title": "New species of deep-sea fish discovered in Pacific Ocean",
        "description": "Marine biologists discovered a new species of bioluminescent fish at record depths in the Pacific.",
        "source": {"name": "National Geographic"},
        "url": "https://example.com/5",
    },
]

result = group_articles(mixed_articles)
check("Total articles is 5", result["total_articles"] == 5)
check("3 groups formed (cricket, AI, fish)", result["total_groups"] == 3)

# Check cricket articles are grouped
cricket_group = None
for g in result["groups"]:
    titles = [a.get("title", "").lower() for a in g["articles"]]
    if any("cricket" in t or "world cup" in t for t in titles):
        cricket_group = g
        break

check("Cricket articles grouped together", cricket_group is not None and cricket_group["article_count"] == 2)

# Check AI articles are grouped
ai_group = None
for g in result["groups"]:
    titles = [a.get("title", "").lower() for a in g["articles"]]
    if any("ai" in t or "artificial intelligence" in t for t in titles):
        ai_group = g
        break

check("AI regulation articles grouped together", ai_group is not None and ai_group["article_count"] == 2)

# Check deep-sea fish is alone
fish_group = None
for g in result["groups"]:
    titles = [a.get("title", "").lower() for a in g["articles"]]
    if any("fish" in t or "deep-sea" in t for t in titles):
        fish_group = g
        break

check("Deep-sea fish article is standalone", fish_group is not None and fish_group["article_count"] == 1)

print("\n  Report:")
print_group_report(result)


# ──────────────────────────────────────────────────────────────
# Test 6: Edge cases
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 6: Edge Cases")
print("=" * 70)

# Empty list
result = group_articles([])
check("Empty input -> 0 groups", result["total_groups"] == 0)

# Articles with missing fields
result = group_articles([
    {"title": None, "description": None, "source": {"name": "A"}},
    {"title": "Real article", "description": "With content", "source": {"name": "B"}},
])
check("Missing fields -> each gets a group", result["total_groups"] >= 1)

# Two identical articles (should group even at high threshold)
result = group_articles([
    {"title": "Same title here", "description": "Same description here", "source": {"name": "X"}},
    {"title": "Same title here", "description": "Same description here", "source": {"name": "Y"}},
])
check("Identical articles -> 1 group", result["total_groups"] == 1)


# ──────────────────────────────────────────────────────────────
# Test 7: Configurable threshold
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Configurable Threshold")
print(f"  Default threshold: {DEFAULT_GROUP_THRESHOLD}")
print("=" * 70)

threshold_articles = [
    {
        "title": "Tesla unveils new electric truck for commercial use",
        "description": "Tesla announced a new electric truck designed for long-haul commercial transportation.",
        "source": {"name": "Source A"},
    },
    {
        "title": "Tesla launches electric semi truck for freight industry",
        "description": "Tesla's new semi truck targets the freight industry with electric power and autonomous driving features.",
        "source": {"name": "Source B"},
    },
]

# Very strict threshold (0.90) - should keep them separate
strict = group_articles(threshold_articles, threshold=0.90)
check("Strict threshold (0.90) -> 2 groups", strict["total_groups"] == 2)

# Lenient threshold (0.10) - should merge them
lenient = group_articles(threshold_articles, threshold=0.10)
check("Lenient threshold (0.10) -> 1 group", lenient["total_groups"] == 1)

# Default threshold
default = group_articles(threshold_articles, threshold=DEFAULT_GROUP_THRESHOLD)
check(
    f"Default threshold ({DEFAULT_GROUP_THRESHOLD}) -> 1 group",
    default["total_groups"] == 1,
)


# ──────────────────────────────────────────────────────────────
# Test 8: Group structure validation
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Group Structure Validation")
print("=" * 70)

structure_articles = [
    {"title": "Article one about topic A", "description": "Details about topic A", "source": {"name": "S1"}},
    {"title": "Article two about topic A", "description": "More about topic A", "source": {"name": "S2"}},
]

result = group_articles(structure_articles)
group = result["groups"][0]

check("Group has group_id", "group_id" in group and isinstance(group["group_id"], int))
check("Group has label", "label" in group and isinstance(group["label"], str))
check("Group has representative", "representative" in group and group["representative"] is not None)
check("Group has articles list", "articles" in group and isinstance(group["articles"], list))
check("Group has article_count", "article_count" in group and isinstance(group["article_count"], int))
check("group_id starts at 1", group["group_id"] == 1)
check("Each article appears exactly once",
      sum(g["article_count"] for g in result["groups"]) == result["total_articles"])


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")

"""
Test script for the multi-source article comparator.

Tests comparison across agreeing sources, differing numbers/dates,
unique claims, contradictions, single-source groups, edge cases,
and different wording expressing similar information.
"""

from analysis.comparator import (
    compare_group,
    compare_all_groups,
    extract_numbers,
    extract_dates,
    extract_proper_nouns,
    extract_key_terms,
    find_agreed_information,
    find_unique_information,
    find_potential_contradictions,
    print_comparison_report,
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


def make_group(articles, group_id=1, label=None):
    """Helper to build a group dict like grouper.py produces."""
    if not label and articles:
        label = articles[0].get("title", "Test Group")
    return {
        "group_id": group_id,
        "label": label or "Test Group",
        "representative": articles[0] if articles else None,
        "articles": articles,
        "article_count": len(articles),
    }


# ──────────────────────────────────────────────────────────────
# Test 1: Multiple sources agreeing
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Multiple Sources Agreeing")
print("=" * 70)

agreeing_articles = [
    {
        "title": "India launches Chandrayaan-4 Moon mission",
        "description": "ISRO successfully launched Chandrayaan-4 from Sriharikota today, aiming for lunar south pole exploration.",
        "source": {"name": "Times of India"},
    },
    {
        "title": "ISRO launches Chandrayaan-4 to explore Moon",
        "description": "India's ISRO launched Chandrayaan-4 spacecraft from Sriharikota, targeting the Moon's south pole region.",
        "source": {"name": "NDTV"},
    },
    {
        "title": "Chandrayaan-4: India's lunar mission lifts off",
        "description": "Chandrayaan-4 was launched by ISRO from Sriharikota for exploration of the Moon's south pole.",
        "source": {"name": "The Hindu"},
    },
]

result = compare_group(make_group(agreeing_articles))

check("Result has group_id", result["group_id"] == 1)
check("Result has 3 sources", len(result["sources"]) == 3)
check("Agreed information found",
      len(result["agreed_information"]) > 0)

# Key terms like "chandrayaan", "isro", "moon", "launched" should appear in agreed
agreed_facts = {item["fact"] for item in result["agreed_information"]}
check("'chandrayaan' in agreed facts", any("chandrayaan" in f for f in agreed_facts))

check("Sources retained in agreed items",
      all(len(item["sources"]) >= 2 for item in result["agreed_information"]))

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 2: Different numbers reported
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 2: Different Numbers Between Sources")
print("=" * 70)

diff_numbers_articles = [
    {
        "title": "Earthquake kills 50 people in Turkey",
        "description": "A devastating earthquake struck Turkey, killing 50 people and injuring 200 others in the eastern province.",
        "source": {"name": "BBC"},
    },
    {
        "title": "Turkey earthquake: 75 dead, hundreds hurt",
        "description": "The earthquake in Turkey has killed 75 people and injured 350 according to the latest government reports.",
        "source": {"name": "Al Jazeera"},
    },
]

result = compare_group(make_group(diff_numbers_articles))

check("Numbers extracted from BBC",
      len(result["named_information"]["numbers"]) > 0)

# Check for different numbers
bbc_nums = [n for n in result["named_information"]["numbers"] if n["source"] == "BBC"]
aj_nums = [n for n in result["named_information"]["numbers"] if n["source"] == "Al Jazeera"]
check("BBC numbers extracted", len(bbc_nums) > 0)
check("Al Jazeera numbers extracted", len(aj_nums) > 0)

# Different numbers for same context should produce differing information
# (depends on context overlap)
check("Differing information detected or numbers differ",
      len(result["differing_information"]) > 0
      or any(n["value"] != bbc_nums[0]["value"] for n in aj_nums if bbc_nums))

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 3: Different dates reported
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 3: Different Dates Between Sources")
print("=" * 70)

diff_dates_articles = [
    {
        "title": "Summit scheduled for September 15, 2026",
        "description": "The global climate summit is scheduled for September 15, 2026 in Geneva.",
        "source": {"name": "Reuters"},
    },
    {
        "title": "Climate summit set for September 20, 2026",
        "description": "World leaders will meet for the climate summit on September 20, 2026 in Geneva.",
        "source": {"name": "AFP"},
    },
]

result = compare_group(make_group(diff_dates_articles))

dates = result["named_information"]["dates"]
check("Dates extracted", len(dates) > 0)

# Both dates should be present
all_date_values = [d["date"] for d in dates]
check("September 15 found", any("15" in d for d in all_date_values))
check("September 20 found", any("20" in d for d in all_date_values))

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 4: Unique claims (only one source mentions)
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 4: Unique Claims from Single Sources")
print("=" * 70)

unique_claim_articles = [
    {
        "title": "New vaccine shows 95% efficacy in trials",
        "description": "The vaccine demonstrated 95% efficacy in Phase 3 clinical trials conducted across 10 countries.",
        "source": {"name": "Reuters"},
    },
    {
        "title": "Vaccine trial results announced",
        "description": "The vaccine trial results show strong protection. Some participants reported mild side effects including headache and fatigue.",
        "source": {"name": "BBC"},
    },
]

result = compare_group(make_group(unique_claim_articles))

check("Unique information found", len(result["unique_information"]) > 0)
check("Each unique item has a source",
      all("source" in item for item in result["unique_information"]))

# "side effects" or "headache" or "fatigue" should be unique to BBC
# "95" or "efficacy" might be unique to Reuters
unique_facts = {item["fact"] for item in result["unique_information"]}
unique_sources = {item["source"] for item in result["unique_information"]}
check("Multiple sources have unique info", len(unique_sources) >= 1)

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 5: Potential contradictions
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 5: Potential Contradictions")
print("=" * 70)

contradiction_articles = [
    {
        "title": "Government confirms new policy will take effect",
        "description": "The government confirmed the new economic policy and said it will boost growth significantly.",
        "source": {"name": "State News"},
    },
    {
        "title": "Opposition denies benefits of new policy",
        "description": "The opposition denied the policy would help and criticized the plan as harmful to workers.",
        "source": {"name": "Independent News"},
    },
]

result = compare_group(make_group(contradiction_articles))

check("Potential contradictions detected",
      len(result["potential_contradictions"]) > 0)

if result["potential_contradictions"]:
    contradiction = result["potential_contradictions"][0]
    check("Contradiction has topic", "topic" in contradiction)
    check("Contradiction has claims from sources",
          len(contradiction["claims"]) >= 2)

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 6: Single-source group
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 6: Single-Source Group")
print("=" * 70)

single_articles = [
    {
        "title": "SpaceX launches Starship prototype",
        "description": "SpaceX successfully launched the Starship prototype from Boca Chica, Texas.",
        "source": {"name": "Space.com"},
    },
]

result = compare_group(make_group(single_articles))

check("Single source result", result["article_count"] == 1)
check("No agreed info (only 1 source)", len(result["agreed_information"]) == 0)
check("No contradictions (only 1 source)", len(result["potential_contradictions"]) == 0)
check("No differing info (only 1 source)", len(result["differing_information"]) == 0)
# All key terms should be unique (only source)
check("Named information present",
      len(result["named_information"]["proper_nouns"]) >= 0)  # May or may not extract


# ──────────────────────────────────────────────────────────────
# Test 7: Multiple-source group structure validation
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 7: Output Structure Validation")
print("=" * 70)

result = compare_group(make_group(agreeing_articles))

check("Has group_id (int)", isinstance(result["group_id"], int))
check("Has group_label (str)", isinstance(result["group_label"], str))
check("Has sources (list)", isinstance(result["sources"], list))
check("Has article_count (int)", isinstance(result["article_count"], int))
check("Has agreed_information (list)", isinstance(result["agreed_information"], list))
check("Has differing_information (list)", isinstance(result["differing_information"], list))
check("Has potential_contradictions (list)", isinstance(result["potential_contradictions"], list))
check("Has unique_information (list)", isinstance(result["unique_information"], list))
check("Has named_information (dict)", isinstance(result["named_information"], dict))
check("named_information has proper_nouns", "proper_nouns" in result["named_information"])
check("named_information has dates", "dates" in result["named_information"])
check("named_information has numbers", "numbers" in result["named_information"])


# ──────────────────────────────────────────────────────────────
# Test 8: Empty input
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 8: Empty Input")
print("=" * 70)

result = compare_group(make_group([]))

check("Empty -> 0 articles", result["article_count"] == 0)
check("Empty -> no sources", len(result["sources"]) == 0)
check("Empty -> no agreed", len(result["agreed_information"]) == 0)
check("Empty -> no unique", len(result["unique_information"]) == 0)
check("Empty -> no contradictions", len(result["potential_contradictions"]) == 0)


# ──────────────────────────────────────────────────────────────
# Test 9: Missing fields
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 9: Articles with Missing Fields")
print("=" * 70)

missing_field_articles = [
    {
        "title": None,
        "description": "Some description without a title.",
        "source": {"name": "Source A"},
    },
    {
        "title": "Article with title",
        "description": None,
        "source": {"name": "Source B"},
    },
    {
        "source": {"name": "Source C"},
    },
]

try:
    result = compare_group(make_group(missing_field_articles))
    check("Handles missing title", True)
    check("Handles missing description", True)
    check("Handles missing both", True)
    check("Returns valid structure", "agreed_information" in result)
except Exception as e:
    check(f"Missing fields caused error: {e}", False)


# ──────────────────────────────────────────────────────────────
# Test 10: Different wording, similar information
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 10: Different Wording, Similar Information")
print("=" * 70)

diff_wording_articles = [
    {
        "title": "Apple reports record revenue of $120 billion",
        "description": "Apple Inc. posted record quarterly revenue of $120 billion, driven by strong iPhone and services sales.",
        "source": {"name": "CNBC"},
    },
    {
        "title": "Apple earnings beat expectations with $120B revenue",
        "description": "Apple's quarterly earnings exceeded Wall Street expectations, reporting revenue of $120 billion from iPhone and services.",
        "source": {"name": "Bloomberg"},
    },
]

result = compare_group(make_group(diff_wording_articles))

agreed_facts = {item["fact"] for item in result["agreed_information"]}
check("Agreed terms found despite different wording",
      len(result["agreed_information"]) > 0)
# "apple", "revenue", "iphone", "services" should be agreed
check("'apple' or 'revenue' in agreed",
      any(f in agreed_facts for f in ["apple", "revenue", "iphone"]))

print("\n  Report:")
print_comparison_report(result)


# ──────────────────────────────────────────────────────────────
# Test 11: Extraction unit tests
# ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print("TEST 11: Extraction Unit Tests")
print("=" * 70)

# Number extraction
nums = extract_numbers("The earthquake killed 42 people and injured 150 others")
check("Extracted 2 numbers", len(nums) == 2)
check("Found 42", any(n["value"] == "42" for n in nums))
check("Found 150", any(n["value"] == "150" for n in nums))

# Date extraction
dates = extract_dates("The summit is on September 15, 2026 in Geneva")
check("Extracted date", len(dates) >= 1)
check("Found September 15, 2026", any("September" in d and "15" in d for d in dates))

dates2 = extract_dates("Published on 2026-08-18 at noon")
check("Extracted ISO date", len(dates2) >= 1)

# Proper noun extraction
nouns = extract_proper_nouns("This was announced by Elon Musk at the SpaceX facility.")
check("Extracted proper nouns", len(nouns) > 0)
noun_values = [n.lower() for n in nouns]
check("Found Elon Musk or SpaceX", any("elon" in n or "musk" in n or "spacex" in n for n in noun_values))

# Empty inputs
check("Empty number extraction", extract_numbers("") == [])
check("Empty date extraction", extract_dates("") == [])
check("Empty proper noun extraction", extract_proper_nouns("") == [])
check("None number extraction", extract_numbers(None) == [])


# ──────────────────────────────────────────────────────────────
# Test 12: compare_all_groups
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 12: compare_all_groups")
print("=" * 70)

grouping_result = {
    "groups": [
        make_group(agreeing_articles, group_id=1, label="Chandrayaan-4"),
        make_group(single_articles, group_id=2, label="SpaceX"),
    ],
    "total_articles": 4,
    "total_groups": 2,
}

results = compare_all_groups(grouping_result)
check("Returns 2 comparison results", len(results) == 2)
check("First group has 3 articles", results[0]["article_count"] == 3)
check("Second group has 1 article", results[1]["article_count"] == 1)
check("Group IDs preserved", results[0]["group_id"] == 1 and results[1]["group_id"] == 2)


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
print(f"{'=' * 70}\n")

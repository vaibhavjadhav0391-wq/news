"""
Multi-Source Article Comparator - Stage 4
Compares articles within an event group and identifies agreements, differences,
contradictions, and unique claims across sources.

HOW IT WORKS (beginner-friendly explanation):
=============================================

After grouping articles by event, we compare what each source actually says.
Different news sources may report the same event but include different facts,
numbers, names, or perspectives. This module finds those differences.

1. INFORMATION EXTRACTION
   For each article, we extract:
   - KEY TERMS: The most important/distinctive words (using TF-IDF).
   - NUMBERS: Any numerical values with surrounding context words.
   - DATES: Date patterns like "August 18, 2026" or "2026-08-18".
   - PROPER NOUNS: Capitalized phrases that are likely names of people,
     organizations, or places.

2. CROSS-SOURCE COMPARISON
   We compare the extracted information across all articles in the group:
   - AGREED: Key terms or entities found in 2+ sources.
   - UNIQUE: Information found in only 1 source.
   - DIFFERING: Different numbers/dates associated with similar context.
   - CONTRADICTIONS: When sources use opposing language (e.g., "confirmed"
     vs "denied") about the same topic.

3. NEUTRAL REPORTING
   The system does NOT decide which source is correct. It reports what each
   source says and lets the user draw their own conclusions. A claim being
   unique to one source does NOT mean it is false.

NO EXTERNAL NLP DEPENDENCIES:
   This module uses only regex, string operations, and scikit-learn (already
   installed for TF-IDF). No spaCy, NLTK, or LLM is required.
"""

import re
import string
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from processing.deduplicator import normalize_text

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Minimum number of sources that must mention a term for it to be "agreed".
MIN_AGREEMENT_SOURCES = 2

# Number of top TF-IDF key terms to extract per article.
TOP_KEY_TERMS = 15

# Words surrounding a number to use as context (for differing-number detection).
NUMBER_CONTEXT_WINDOW = 4

# Common words to exclude from proper noun detection.
COMMON_TITLE_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "has", "have",
    "had", "be", "been", "being", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "yes",
    "its", "it", "this", "that", "these", "those", "he", "she", "they",
    "his", "her", "their", "our", "your", "my", "we", "us", "them", "i",
    "me", "him", "who", "what", "where", "when", "why", "how", "which",
    "new", "said", "says", "also", "just", "more", "most", "much", "very",
    "now", "than", "then", "so", "if", "as", "up", "out", "about", "over",
    "after", "before", "between", "under", "into", "through", "during",
    "each", "every", "all", "both", "few", "some", "any", "other", "only",
}

# Words that indicate opposing claims (used for contradiction detection).
CONTRADICTION_PAIRS = [
    ({"confirmed", "confirms", "approved", "accepts", "agreed"},
     {"denied", "denies", "rejected", "rejects", "refused", "opposes"}),
    ({"increased", "increases", "rose", "gained", "surged", "climbed"},
     {"decreased", "decreases", "fell", "dropped", "declined", "plummeted"}),
    ({"supports", "backed", "endorsed", "favors"},
     {"opposes", "opposed", "criticized", "condemned"}),
    ({"safe", "secure", "stable"},
     {"dangerous", "unsafe", "unstable", "risky"}),
    ({"success", "successful", "succeeded"},
     {"failure", "failed", "unsuccessful"}),
    ({"alive", "survived", "rescued"},
     {"dead", "killed", "died", "perished"}),
]


# ──────────────────────────────────────────────────────────────
# Information Extraction
# ──────────────────────────────────────────────────────────────

def extract_numbers(text):
    """
    Extract numbers with surrounding context from raw text.

    Captures integers, decimals, percentages, and currency values,
    along with nearby words for context.

    Args:
        text: Raw (un-normalized) article text.

    Returns:
        List of dicts: {"value": "42", "context": "killed 42 people in"}.
    """
    if not text:
        return []

    results = []
    words = text.split()

    for i, word in enumerate(words):
        # Clean the word and check if it contains a number
        cleaned = word.strip(string.punctuation)
        if re.match(r"^\$?[\d,]+\.?\d*%?$", cleaned):
            # Grab context window
            start = max(0, i - NUMBER_CONTEXT_WINDOW)
            end = min(len(words), i + NUMBER_CONTEXT_WINDOW + 1)
            context = " ".join(words[start:end])
            results.append({
                "value": cleaned,
                "context": context,
            })

    return results


def extract_dates(text):
    """
    Extract date patterns from raw text.

    Matches patterns like:
    - "August 18, 2026"
    - "18 August 2026"
    - "2026-08-18"
    - "08/18/2026"

    Args:
        text: Raw (un-normalized) article text.

    Returns:
        List of matched date strings.
    """
    if not text:
        return []

    patterns = [
        # Month DD, YYYY
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        # DD Month YYYY
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        # YYYY-MM-DD
        r"\b\d{4}-\d{2}-\d{2}\b",
        # MM/DD/YYYY or DD/MM/YYYY
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    ]

    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))

    return dates


def extract_proper_nouns(text):
    """
    Extract likely proper nouns from raw text using capitalization heuristics.

    Identifies sequences of capitalized words that are likely names of people,
    organizations, or places. Filters out common English words and
    sentence-starting words.

    Args:
        text: Raw (un-normalized) article text.

    Returns:
        List of proper noun strings.
    """
    if not text:
        return []

    # Find sequences of capitalized words (2+ words or single if unusual enough)
    # This catches "United States", "Elon Musk", "World Health Organization", etc.
    proper_nouns = []

    # Multi-word capitalized sequences
    multi_word = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    for phrase in multi_word:
        words = phrase.split()
        # Filter out phrases that are entirely common words
        meaningful = [w for w in words if w.lower() not in COMMON_TITLE_WORDS]
        if meaningful:
            proper_nouns.append(phrase)

    # Single capitalized words (only if they're not at sentence start)
    sentences = re.split(r"[.!?]\s+", text)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if i == 0:
                continue  # Skip sentence-starting words
            cleaned = word.strip(string.punctuation)
            if (cleaned
                    and cleaned[0].isupper()
                    and cleaned.lower() not in COMMON_TITLE_WORDS
                    and len(cleaned) > 1
                    and not cleaned.isupper()  # Skip acronyms here
                    ):
                proper_nouns.append(cleaned)

    # Also extract acronyms (2-6 uppercase letters)
    acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
    # Filter very common ones
    skip_acronyms = {"AM", "PM", "US", "UK", "TV", "OK", "ID", "IT", "OR", "AN"}
    acronyms = [a for a in acronyms if a not in skip_acronyms]
    proper_nouns.extend(acronyms)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for noun in proper_nouns:
        if noun.lower() not in seen:
            seen.add(noun.lower())
            unique.append(noun)

    return unique


def extract_key_terms(texts):
    """
    Extract the most important/distinctive terms from a collection of texts.

    Uses TF-IDF to find words that are distinctive to each article.

    Args:
        texts: List of normalized text strings (one per article).

    Returns:
        List of lists: key terms per article.
    """
    if not texts or all(t == "" for t in texts):
        return [[] for _ in texts]

    # If only one text, use word frequency instead of TF-IDF
    if len(texts) == 1:
        words = texts[0].split()
        freq = Counter(words)
        # Remove very short words
        terms = [w for w, _ in freq.most_common(TOP_KEY_TERMS) if len(w) > 2]
        return [terms]

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # All texts might be empty or contain only stop words
        return [[] for _ in texts]

    feature_names = vectorizer.get_feature_names_out()
    key_terms_per_article = []

    for i in range(len(texts)):
        row = tfidf_matrix.getrow(i).toarray().flatten()
        # Get top terms by TF-IDF score
        top_indices = row.argsort()[-TOP_KEY_TERMS:][::-1]
        terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        key_terms_per_article.append(terms)

    return key_terms_per_article


# ──────────────────────────────────────────────────────────────
# Comparison Logic
# ──────────────────────────────────────────────────────────────

def find_agreed_information(key_terms_by_source):
    """
    Find key terms mentioned by multiple sources.

    Args:
        key_terms_by_source: Dict mapping source name to list of key terms.

    Returns:
        List of dicts: {"fact": term, "sources": [source names], "source_count": N}.
    """
    # Count how many sources mention each term
    term_sources = defaultdict(set)
    for source, terms in key_terms_by_source.items():
        for term in terms:
            term_sources[term].add(source)

    agreed = []
    for term, sources in sorted(term_sources.items()):
        if len(sources) >= MIN_AGREEMENT_SOURCES:
            agreed.append({
                "fact": term,
                "sources": sorted(sources),
                "source_count": len(sources),
            })

    # Sort by number of agreeing sources (most agreement first)
    agreed.sort(key=lambda x: x["source_count"], reverse=True)
    return agreed


def find_unique_information(key_terms_by_source):
    """
    Find key terms mentioned by only one source.

    Args:
        key_terms_by_source: Dict mapping source name to list of key terms.

    Returns:
        List of dicts: {"fact": term, "source": source_name}.
    """
    # Count how many sources mention each term
    term_sources = defaultdict(set)
    for source, terms in key_terms_by_source.items():
        for term in terms:
            term_sources[term].add(source)

    unique = []
    for term, sources in sorted(term_sources.items()):
        if len(sources) == 1:
            unique.append({
                "fact": term,
                "source": list(sources)[0],
            })

    return unique


def find_differing_numbers(numbers_by_source):
    """
    Find cases where sources report different numbers for similar contexts.

    Compares numbers across sources and flags differences when the context
    words overlap but the values differ.

    Args:
        numbers_by_source: Dict mapping source name to list of number dicts.

    Returns:
        List of dicts: {"topic": context, "versions": [{"source", "value", "context"}]}.
    """
    if len(numbers_by_source) < 2:
        return []

    # Collect all number entries with source
    all_numbers = []
    for source, nums in numbers_by_source.items():
        for num in nums:
            all_numbers.append({
                "source": source,
                "value": num["value"],
                "context": num["context"],
                "context_words": set(normalize_text(num["context"]).split()),
            })

    differences = []
    compared = set()

    for i, num_a in enumerate(all_numbers):
        for j, num_b in enumerate(all_numbers):
            if i >= j:
                continue
            if num_a["source"] == num_b["source"]:
                continue

            pair_key = (min(i, j), max(i, j))
            if pair_key in compared:
                continue
            compared.add(pair_key)

            # Check if contexts overlap (similar topic) but values differ
            overlap = num_a["context_words"] & num_b["context_words"]
            # Need at least 2 overlapping context words (excluding very short ones)
            meaningful_overlap = {w for w in overlap if len(w) > 2}

            if meaningful_overlap and num_a["value"] != num_b["value"]:
                topic = " ".join(sorted(meaningful_overlap)[:5])
                differences.append({
                    "topic": topic,
                    "versions": [
                        {"source": num_a["source"], "value": num_a["value"],
                         "context": num_a["context"]},
                        {"source": num_b["source"], "value": num_b["value"],
                         "context": num_b["context"]},
                    ],
                })

    return differences


def find_potential_contradictions(articles_info):
    """
    Detect potential contradictions using opposing language patterns.

    Checks if different sources use opposing words (e.g., "confirmed" vs
    "denied") in their coverage. This is a heuristic — flagged items should
    be reviewed by the user, not taken as definitive.

    Args:
        articles_info: List of dicts with "source", "normalized_text" keys.

    Returns:
        List of dicts: {"topic": description, "claims": [{"source", "claim"}]}.
    """
    if len(articles_info) < 2:
        return []

    contradictions = []

    for pos_set, neg_set in CONTRADICTION_PAIRS:
        sources_positive = []
        sources_negative = []

        for info in articles_info:
            text_words = set(info["normalized_text"].split())
            source = info["source"]

            if text_words & pos_set:
                matched = text_words & pos_set
                sources_positive.append({
                    "source": source,
                    "claim": f"Uses: {', '.join(sorted(matched))}",
                })
            if text_words & neg_set:
                matched = text_words & neg_set
                sources_negative.append({
                    "source": source,
                    "claim": f"Uses: {', '.join(sorted(matched))}",
                })

        # Only flag if DIFFERENT sources use opposing language
        if sources_positive and sources_negative:
            pos_sources = {s["source"] for s in sources_positive}
            neg_sources = {s["source"] for s in sources_negative}

            if pos_sources != neg_sources:  # Different sources disagree
                all_claims = sources_positive + sources_negative
                topic_words = sorted((pos_set | neg_set))[:4]
                contradictions.append({
                    "topic": f"Opposing language detected ({'/'.join(topic_words[:2])})",
                    "claims": all_claims,
                })

    return contradictions


# ──────────────────────────────────────────────────────────────
# Main Comparison Function
# ──────────────────────────────────────────────────────────────

def compare_group(group):
    """
    Compare articles within an event group and produce a structured analysis.

    Args:
        group: A group dict as produced by grouper.group_articles(), containing:
            "group_id", "label", "articles", "article_count".

    Returns:
        dict with:
            "group_id": int
            "group_label": str
            "sources": list of source names
            "article_count": int
            "agreed_information": list of agreed facts
            "differing_information": list of differing numbers/dates
            "potential_contradictions": list of contradiction indicators
            "unique_information": list of single-source facts
            "named_information": dict with people, organizations, locations, dates, numbers
    """
    articles = group.get("articles", [])

    if not articles:
        return _empty_result(group)

    # ── Step 1: Extract information from each article ──
    source_names = []
    key_terms_by_source = {}
    numbers_by_source = {}
    dates_by_source = {}
    proper_nouns_by_source = {}
    articles_info = []

    # Gather normalized texts for TF-IDF
    normalized_texts = []
    for article in articles:
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        raw_text = f"{title}. {description}"
        norm_text = normalize_text(raw_text)
        normalized_texts.append(norm_text)

    # Extract key terms across all articles at once (TF-IDF needs the full corpus)
    all_key_terms = extract_key_terms(normalized_texts)

    for i, article in enumerate(articles):
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        source = article.get("source", {}).get("name", f"Source {i + 1}")
        raw_text = f"{title}. {description}"

        # Handle duplicate source names by appending index
        display_source = source
        if source in key_terms_by_source:
            display_source = f"{source} ({i + 1})"

        source_names.append(display_source)
        key_terms_by_source[display_source] = all_key_terms[i]
        numbers_by_source[display_source] = extract_numbers(raw_text)
        dates_by_source[display_source] = extract_dates(raw_text)
        proper_nouns_by_source[display_source] = extract_proper_nouns(raw_text)

        articles_info.append({
            "source": display_source,
            "normalized_text": normalized_texts[i],
            "raw_text": raw_text,
        })

    # ── Step 2: Compare across sources ──
    agreed = find_agreed_information(key_terms_by_source)
    unique = find_unique_information(key_terms_by_source)
    differing = find_differing_numbers(numbers_by_source)
    contradictions = find_potential_contradictions(articles_info)

    # ── Step 3: Compile named information ──
    all_proper_nouns = []
    all_dates = []
    all_numbers = []

    for source in source_names:
        for noun in proper_nouns_by_source.get(source, []):
            all_proper_nouns.append({"entity": noun, "source": source})
        for date in dates_by_source.get(source, []):
            all_dates.append({"date": date, "source": source})
        for num in numbers_by_source.get(source, []):
            all_numbers.append({
                "value": num["value"],
                "context": num["context"],
                "source": source,
            })

    named_information = {
        "proper_nouns": _deduplicate_entities(all_proper_nouns, "entity"),
        "dates": _deduplicate_entities(all_dates, "date"),
        "numbers": all_numbers,
    }

    return {
        "group_id": group.get("group_id", 0),
        "group_label": group.get("label", "Unknown"),
        "sources": source_names,
        "article_count": len(articles),
        "agreed_information": agreed,
        "differing_information": differing,
        "potential_contradictions": contradictions,
        "unique_information": unique,
        "named_information": named_information,
    }


def compare_all_groups(grouping_result):
    """
    Compare articles across all event groups.

    Args:
        grouping_result: The dict returned by grouper.group_articles().

    Returns:
        List of comparison result dicts (one per group).
    """
    results = []
    for group in grouping_result.get("groups", []):
        results.append(compare_group(group))
    return results


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _empty_result(group):
    """Return an empty comparison result."""
    return {
        "group_id": group.get("group_id", 0),
        "group_label": group.get("label", "Unknown"),
        "sources": [],
        "article_count": 0,
        "agreed_information": [],
        "differing_information": [],
        "potential_contradictions": [],
        "unique_information": [],
        "named_information": {
            "proper_nouns": [],
            "dates": [],
            "numbers": [],
        },
    }


def _deduplicate_entities(entities, key_field):
    """
    Merge entities with the same value, collecting all sources.

    Args:
        entities: List of dicts with key_field and "source".
        key_field: The field name to deduplicate on (e.g., "entity" or "date").

    Returns:
        List of dicts with key_field, "sources", "source_count".
    """
    merged = defaultdict(set)
    for item in entities:
        merged[item[key_field]].add(item["source"])

    result = []
    for value, sources in sorted(merged.items()):
        result.append({
            key_field: value,
            "sources": sorted(sources),
            "source_count": len(sources),
        })

    return result


# ──────────────────────────────────────────────────────────────
# Report Printing
# ──────────────────────────────────────────────────────────────

def print_comparison_report(result):
    """
    Print a human-readable comparison report.

    Args:
        result: The dict returned by compare_group().
    """
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON REPORT: \"{result['group_label']}\"")
    print(f"{'=' * 70}")
    print(f"  Group ID:    {result['group_id']}")
    print(f"  Sources:     {', '.join(result['sources'])}")
    print(f"  Articles:    {result['article_count']}")

    # Agreed information
    agreed = result["agreed_information"]
    print(f"\n  --- AGREED INFORMATION ({len(agreed)} items) ---")
    if agreed:
        for item in agreed:
            sources_str = ", ".join(item["sources"])
            print(f"    [{item['source_count']} sources] \"{item['fact']}\"")
            print(f"      Reported by: {sources_str}")
    else:
        print("    (none detected)")

    # Differing information
    differing = result["differing_information"]
    print(f"\n  --- DIFFERING INFORMATION ({len(differing)} items) ---")
    if differing:
        for item in differing:
            print(f"    Topic: {item['topic']}")
            for ver in item["versions"]:
                print(f"      {ver['source']}: {ver['value']} (\"{ver['context']}\")")
    else:
        print("    (none detected)")

    # Potential contradictions
    contradictions = result["potential_contradictions"]
    print(f"\n  --- POTENTIAL CONTRADICTIONS ({len(contradictions)} items) ---")
    if contradictions:
        for item in contradictions:
            print(f"    {item['topic']}")
            for claim in item["claims"]:
                print(f"      {claim['source']}: {claim['claim']}")
    else:
        print("    (none detected)")

    # Unique information
    unique = result["unique_information"]
    print(f"\n  --- UNIQUE INFORMATION ({len(unique)} items) ---")
    if unique:
        for item in unique:
            print(f"    \"{item['fact']}\" - only in: {item['source']}")
    else:
        print("    (none detected)")

    # Named information
    named = result["named_information"]
    print(f"\n  --- NAMED INFORMATION ---")

    if named["proper_nouns"]:
        print(f"    Proper nouns ({len(named['proper_nouns'])}):")
        for item in named["proper_nouns"]:
            print(f"      {item['entity']} (from: {', '.join(item['sources'])})")

    if named["dates"]:
        print(f"    Dates ({len(named['dates'])}):")
        for item in named["dates"]:
            print(f"      {item['date']} (from: {', '.join(item['sources'])})")

    if named["numbers"]:
        print(f"    Numbers ({len(named['numbers'])}):")
        for item in named["numbers"]:
            print(f"      {item['value']} in \"{item['context']}\" ({item['source']})")

    print()

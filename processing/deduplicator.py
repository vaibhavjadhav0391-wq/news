"""
Article Deduplicator - Stage 2
Detects and removes duplicate or near-duplicate news articles.

HOW IT WORKS (beginner-friendly explanation):
=============================================

When we search for news about a topic, different sources often publish the
same story — sometimes with identical text, sometimes with slightly different
wording. We want to remove these duplicates so the user sees each unique
story only once.

Here's the step-by-step process:

1. TEXT NORMALIZATION
   We take each article's title and description, then "clean" the text:
   - Convert to lowercase ("Apple" → "apple")
   - Remove extra spaces ("hello   world" → "hello world")
   - Remove punctuation ("it's great!" → "its great")
   This ensures small formatting differences don't fool the comparison.

2. TF-IDF VECTORIZATION
   We convert each article's cleaned text into a list of numbers (a "vector").
   TF-IDF stands for "Term Frequency - Inverse Document Frequency":
   - Words that appear often in ONE article but rarely in OTHERS get high scores.
   - Common words like "the", "is", "and" get low scores automatically.
   This turns text into numbers that a computer can compare mathematically.

3. COSINE SIMILARITY
   We compare every pair of article vectors by measuring how similar their
   "directions" are. The result is a number between 0 and 1:
   - 1.0 = identical text
   - 0.8+ = very similar (likely duplicates)
   - 0.5 = somewhat related
   - 0.0 = completely different
   Think of it like comparing the "angle" between two arrows — if they point
   in the same direction, the articles talk about the same thing.

4. GROUPING DUPLICATES
   Any two articles with similarity above our threshold (default: 0.65) are
   considered duplicates. We keep the first article from each group as the
   "representative" and record the rest as duplicates.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default similarity threshold: articles scoring above this are considered duplicates.
# Range: 0.0 (nothing is a duplicate) to 1.0 (only exact matches).
# 0.65 is a good starting point for news articles.
DEFAULT_SIMILARITY_THRESHOLD = 0.65


def normalize_text(text):
    """
    Clean and normalize text for comparison.

    Steps:
        1. Convert to lowercase.
        2. Remove punctuation.
        3. Collapse multiple spaces into one.
        4. Strip leading/trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse multiple whitespace into single spaces
    text = re.sub(r"\s+", " ", text)

    # Strip edges
    text = text.strip()

    return text


def get_article_text(article):
    """
    Extract and combine the comparable text from an article.

    We use both title and description to get a richer signal for comparison.
    Title alone is too short; description adds context.

    Args:
        article: Article dict (as returned by NewsAPI / news_fetcher).

    Returns:
        Combined normalized text string.
    """
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    combined = f"{title} {description}"
    return normalize_text(combined)


def compute_similarity_matrix(texts):
    """
    Convert texts to TF-IDF vectors and compute pairwise cosine similarity.

    Args:
        texts: List of normalized text strings.

    Returns:
        2D numpy array where entry [i][j] is the similarity between text i and j.
    """
    if not texts or all(t == "" for t in texts):
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


def find_duplicates(articles, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Identify duplicate or near-duplicate articles.

    Args:
        articles: List of article dicts from NewsAPI.
        threshold: Similarity score above which two articles are considered
                   duplicates (0.0 to 1.0). Default: 0.65.

    Returns:
        dict with:
            "unique_articles": list of deduplicated articles (one per group)
            "duplicate_groups": list of dicts, each containing:
                "representative": the kept article
                "duplicates": list of articles that matched it
                "similarity_scores": list of similarity scores for each duplicate
            "total_input": number of articles received
            "total_unique": number of unique articles after deduplication
            "total_removed": number of duplicate articles removed
    """
    if not articles:
        return {
            "unique_articles": [],
            "duplicate_groups": [],
            "total_input": 0,
            "total_unique": 0,
            "total_removed": 0,
        }

    # Step 1: Extract and normalize text from each article
    texts = [get_article_text(article) for article in articles]

    # Step 2: Compute similarity matrix
    sim_matrix = compute_similarity_matrix(texts)

    if len(sim_matrix) == 0:
        return {
            "unique_articles": list(articles),
            "duplicate_groups": [],
            "total_input": len(articles),
            "total_unique": len(articles),
            "total_removed": 0,
        }

    # Step 3: Group duplicates
    # Track which articles have already been claimed as duplicates
    n = len(articles)
    is_duplicate = [False] * n
    duplicate_groups = []

    for i in range(n):
        if is_duplicate[i]:
            continue

        # Find all articles similar to article i
        group_duplicates = []
        group_scores = []

        for j in range(i + 1, n):
            if is_duplicate[j]:
                continue

            score = sim_matrix[i][j]
            if score >= threshold:
                group_duplicates.append(articles[j])
                group_scores.append(round(float(score), 4))
                is_duplicate[j] = True

        if group_duplicates:
            duplicate_groups.append({
                "representative": articles[i],
                "duplicates": group_duplicates,
                "similarity_scores": group_scores,
            })

    # Step 4: Collect unique articles (non-duplicates)
    unique_articles = [
        articles[i] for i in range(n) if not is_duplicate[i]
    ]

    total_removed = sum(1 for d in is_duplicate if d)

    return {
        "unique_articles": unique_articles,
        "duplicate_groups": duplicate_groups,
        "total_input": n,
        "total_unique": len(unique_articles),
        "total_removed": total_removed,
    }


def print_dedup_report(result):
    """
    Print a human-readable deduplication report.

    Args:
        result: The dict returned by find_duplicates().
    """
    print(f"\n{'=' * 70}")
    print("  DEDUPLICATION REPORT")
    print(f"{'=' * 70}")
    print(f"  Articles received:  {result['total_input']}")
    print(f"  Unique articles:    {result['total_unique']}")
    print(f"  Duplicates removed: {result['total_removed']}")
    print(f"{'=' * 70}")

    if result["duplicate_groups"]:
        print("\n  DUPLICATE GROUPS FOUND:")
        for g_idx, group in enumerate(result["duplicate_groups"], 1):
            rep_title = group["representative"].get("title", "N/A")
            rep_source = group["representative"].get("source", {}).get("name", "N/A")
            print(f"\n  Group {g_idx}:")
            print(f"    Kept:  \"{rep_title}\" ({rep_source})")

            for dup, score in zip(group["duplicates"], group["similarity_scores"]):
                dup_title = dup.get("title", "N/A")
                dup_source = dup.get("source", {}).get("name", "N/A")
                print(f"    Dupe:  \"{dup_title}\" ({dup_source})")
                print(f"           Similarity: {score:.2%}")

    print()

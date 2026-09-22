"""
Development-only Diagnostic Script: Live NewsAPI Relevance Filter Validation

Purpose:
  Validate that Stage 1B Strict Relevance Filter correctly filters real NewsAPI
  responses for target queries (e.g. "India AI regulations", "OpenAI latest model").

SECURITY & PRIVACY:
  - NEVER prints or logs API keys (NEWS_API_KEY, GEMINI_API_KEY).
  - DOES NOT call Gemini (avoids API quota usage and keeps test isolated to fetcher + filter).

Usage:
  python scripts/test_relevance_live.py ["optional custom query"]
"""

import os
import sys
from dotenv import load_dotenv

# Ensure project root is in Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

from news_fetcher import fetch_articles
from processing.relevance_filter import (
    calculate_relevance_score,
    filter_relevant_articles,
    DEFAULT_RELEVANCE_THRESHOLD,
)


def run_diagnostic(query: str, threshold: float = DEFAULT_RELEVANCE_THRESHOLD):
    """Run fetcher + relevance filter on a live query and print diagnostic output."""
    print("=" * 80)
    print(f"DIAGNOSTIC TEST QUERY: \"{query}\" (Threshold = {threshold})")
    print("=" * 80)

    # Security check: Ensure key exists without printing its value
    key = os.getenv("NEWS_API_KEY")
    if not key or key == "YOUR_NEWS_API_KEY":
        print("\n[ERROR] NEWS_API_KEY is missing or set to placeholder in environment/.env.")
        print("Please configure a valid NEWS_API_KEY in .env to run live diagnostics.\n")
        return

    print("\n1. Fetching articles from NewsAPI...")
    try:
        articles = fetch_articles(query)
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch articles: {e}")
        return

    print(f"   -> Retrieved {len(articles)} raw articles from NewsAPI.")

    if not articles:
        print("   -> No articles returned by NewsAPI for this query.")
        return

    print("\n2. Evaluating Relevance Filter Scoring & Decision Breakdown:")
    print("-" * 80)

    kept_count = 0
    rejected_count = 0

    for i, article in enumerate(articles, 1):
        title = article.get("title") or "[No Title]"
        source_name = (
            article.get("source", {}).get("name")
            if isinstance(article.get("source"), dict)
            else "Unknown"
        )
        url = article.get("url") or "[No URL]"

        scored_item = calculate_relevance_score(query, article, threshold=threshold)
        score = scored_item["score"]
        is_relevant = scored_item["passed"]
        rejection_reason = scored_item.get("rejection_reason")

        if is_relevant:
            kept_count += 1
            status_str = f"\033[92m[KEEP]\033[0m (Score: {score:.3f})"
        else:
            rejected_count += 1
            reason_str = f" - Reason: {rejection_reason}" if rejection_reason else ""
            status_str = f"\033[91m[REJECT]\033[0m (Score: {score:.3f}){reason_str}"

        print(f"[{i:02d}] {status_str}")
        print(f"     Title  : {title}")
        print(f"     Source : {source_name}")
        print(f"     URL    : {url}")
        print("-" * 80)

    print("\n3. Diagnostic Summary:")
    print(f"   - Total Articles Retrieved : {len(articles)}")
    print(f"   - Kept (Relevant)          : {kept_count}")
    print(f"   - Rejected (Filtered Out)  : {rejected_count}")
    print("=" * 80)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_diagnostic(query)
    else:
        # Standard benchmark queries
        test_queries = [
            "India AI regulations",
            "OpenAI latest model",
            "India GDP growth",
            "Apple iPhone launch",
        ]

        print("\nRunning benchmark relevance filter diagnostics on standard queries...\n")
        for q in test_queries:
            run_diagnostic(q)
            print("\n")


if __name__ == "__main__":
    main()

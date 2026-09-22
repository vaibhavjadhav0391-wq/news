"""
News Fetcher - Stage 1
Fetches current news articles from NewsAPI based on a user-provided topic.

Usage:
    python news_fetcher.py
"""

import sys
from datetime import datetime
import requests
from config import NEWS_API_KEY, validate_config

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"


def fetch_articles(topic, page_size=30):
    """
    Fetch news articles from NewsAPI for a given topic.

    Args:
        topic: Search query string.
        page_size: Number of articles to retrieve (max 100 for free tier).

    Returns:
        List of article dictionaries, or an empty list on error.
    """
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "relevance",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get(NEWSAPI_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print("Network error: Could not connect to NewsAPI. Check your internet connection.")
        return []
    except requests.exceptions.Timeout:
        print("Network error: Request to NewsAPI timed out. Try again later.")
        return []
    except requests.exceptions.RequestException as e:
        # Redact API key from error messages
        error_msg = str(e)
        if NEWS_API_KEY:
            error_msg = error_msg.replace(NEWS_API_KEY, "[REDACTED]")
        print(f"Error fetching articles: {error_msg}")
        return []

    if data.get("status") != "ok":
        error_msg = data.get("message", "Unknown error")
        if NEWS_API_KEY:
            error_msg = error_msg.replace(NEWS_API_KEY, "[REDACTED]")
        print(f"API error: {error_msg}")
        return []

    return data.get("articles", [])


def display_articles(articles, topic):
    """
    Display articles in a formatted, readable layout.

    Args:
        articles: List of article dicts from NewsAPI.
        topic: The search topic (used in the header).
    """
    if not articles:
        print(f'\nNo articles found for "{topic}".')
        return

    print(f'\n{"=" * 70}')
    print(f'  NEWS RESULTS FOR: "{topic.upper()}"')
    print(f'  Found {len(articles)} article(s)')
    print(f'{"=" * 70}')

    for i, article in enumerate(articles, 1):
        title = article.get("title", "No title")
        source = article.get("source", {}).get("name", "Unknown source")
        author = article.get("author", "Unknown author")
        description = article.get("description", "No description available")
        url = article.get("url", "No URL")

        # Parse and format the publication date
        published_raw = article.get("publishedAt", "")
        if published_raw:
            try:
                dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                published = dt.strftime("%B %d, %Y at %I:%M %p UTC")
            except ValueError:
                published = published_raw
        else:
            published = "Unknown date"

        print(f"\n  [{i}] {title}")
        print(f"      Source:    {source}")
        print(f"      Author:   {author}")
        print(f"      Date:     {published}")
        print(f"      Description: {description}")
        print(f"      URL:      {url}")
        print(f"      {'-' * 60}")

    print()


def main():
    """Main entry point: validate config, get topic, fetch and display."""
    validate_config()

    topic = input("\nEnter a news topic or claim: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        sys.exit(0)

    print(f'\nSearching NewsAPI for "{topic}"...')
    articles = fetch_articles(topic)
    display_articles(articles, topic)


if __name__ == "__main__":
    main()

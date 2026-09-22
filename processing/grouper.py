"""
Event-Based Article Grouper - Stage 3
Groups different articles that report on the same real-world event or story.

HOW IT WORKS (beginner-friendly explanation):
=============================================

After removing duplicates, we still have many articles about the same event
written by different journalists. For example, three different articles about
"India's Mars mission launch" are not duplicates (they use different words),
but they all cover the same event. We want to group them together.

Here's the step-by-step process:

1. TEXT PREPARATION
   We take each article's title and description and normalize them
   (lowercase, remove punctuation, collapse whitespace) — the same
   cleaning used by the deduplicator.

2. TF-IDF VECTORIZATION
   Each article's text is converted into a numerical vector using TF-IDF.
   This captures *what* each article talks about, weighting words that are
   distinctive to a particular article higher than common words.

3. COSINE SIMILARITY
   We compute how similar every pair of articles is. The result is a number
   between 0.0 (completely different) and 1.0 (identical).

4. GROUPING (Single-Pass Clustering)
   We walk through the articles one by one:
     - For the first article, create Group 1.
     - For each subsequent article, compare it against every existing group.
     - If the article is similar enough to ANY article in a group (above the
       threshold), add it to that group.
     - If it doesn't match any group, create a new group for it.

   This is called "single-linkage" clustering — an article joins a group if
   it's close to at least one member of that group.

5. GROUP LABELS
   Each group gets a human-readable label based on the "most representative"
   article — the one whose average similarity to all other group members is
   highest. Its title becomes the group label.

KEY DIFFERENCE FROM DEDUPLICATION:
   - Deduplication removes articles that say nearly the SAME thing (high threshold ~0.65).
   - Event grouping clusters articles about the SAME topic (lower threshold ~0.30).
   Two articles can be in the same event group without being duplicates.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from processing.deduplicator import normalize_text

# Default grouping threshold: articles scoring above this are grouped together.
# Lower than deduplication because we want to catch "same event, different wording".
# Range: 0.0 (everything in one group) to 1.0 (only identical articles grouped).
# 0.30 is a good starting point for news event grouping.
DEFAULT_GROUP_THRESHOLD = 0.30


def get_article_text(article):
    """
    Extract and combine comparable text from an article.

    Uses title + description for a richer comparison signal.

    Args:
        article: Article dict (as returned by NewsAPI / news_fetcher).

    Returns:
        Combined normalized text string.
    """
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    combined = f"{title} {description}"
    return normalize_text(combined)


def find_representative(group_indices, sim_matrix):
    """
    Find the most representative article in a group.

    The representative is the article with the highest average similarity
    to all other members of the group — essentially the "center" of the group.

    Args:
        group_indices: List of article indices belonging to this group.
        sim_matrix: Full pairwise similarity matrix.

    Returns:
        Index of the most representative article.
    """
    if len(group_indices) == 1:
        return group_indices[0]

    best_idx = group_indices[0]
    best_avg = -1.0

    for i in group_indices:
        # Average similarity to all OTHER members in this group
        total = sum(sim_matrix[i][j] for j in group_indices if j != i)
        avg = total / (len(group_indices) - 1)
        if avg > best_avg:
            best_avg = avg
            best_idx = i

    return best_idx


def group_articles(articles, threshold=DEFAULT_GROUP_THRESHOLD):
    """
    Group articles by the real-world event they report on.

    Uses single-linkage clustering: an article joins a group if it is
    sufficiently similar to at least one article already in that group.

    Args:
        articles: List of article dicts (already deduplicated).
        threshold: Similarity score above which two articles are considered
                   to be about the same event (0.0 to 1.0). Default: 0.30.

    Returns:
        dict with:
            "groups": list of group dicts, each containing:
                "group_id": integer ID starting at 1
                "label": human-readable label from representative title
                "representative": the most central article dict
                "articles": list of all article dicts in this group
                "article_count": number of articles in the group
            "total_articles": number of articles received
            "total_groups": number of groups formed
    """
    if not articles:
        return {
            "groups": [],
            "total_articles": 0,
            "total_groups": 0,
        }

    # Single article — one group
    if len(articles) == 1:
        return {
            "groups": [{
                "group_id": 1,
                "label": articles[0].get("title", "Untitled"),
                "representative": articles[0],
                "articles": [articles[0]],
                "article_count": 1,
            }],
            "total_articles": 1,
            "total_groups": 1,
        }

    # Step 1: Extract and normalize text
    texts = [get_article_text(a) for a in articles]

    # Handle case where all texts are empty
    if all(t == "" for t in texts):
        groups = []
        for idx, article in enumerate(articles):
            groups.append({
                "group_id": idx + 1,
                "label": article.get("title", "Untitled"),
                "representative": article,
                "articles": [article],
                "article_count": 1,
            })
        return {
            "groups": groups,
            "total_articles": len(articles),
            "total_groups": len(articles),
        }

    # Step 2: Compute TF-IDF and similarity matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Step 3: Single-linkage clustering
    # Each element is a list of article indices
    clusters = []

    for i in range(len(articles)):
        merged = False

        for cluster in clusters:
            # Check if article i is similar to ANY article in this cluster
            for member_idx in cluster:
                if sim_matrix[i][member_idx] >= threshold:
                    cluster.append(i)
                    merged = True
                    break

            if merged:
                break

        if not merged:
            # Create a new cluster for this article
            clusters.append([i])

    # Step 4: Build group output with labels
    groups = []
    for g_id, cluster_indices in enumerate(clusters, 1):
        rep_idx = find_representative(cluster_indices, sim_matrix)
        rep_article = articles[rep_idx]

        groups.append({
            "group_id": g_id,
            "label": rep_article.get("title", "Untitled"),
            "representative": rep_article,
            "articles": [articles[idx] for idx in cluster_indices],
            "article_count": len(cluster_indices),
        })

    return {
        "groups": groups,
        "total_articles": len(articles),
        "total_groups": len(groups),
    }


def print_group_report(result):
    """
    Print a human-readable event grouping report.

    Args:
        result: The dict returned by group_articles().
    """
    print(f"\n{'=' * 70}")
    print("  EVENT GROUPING REPORT")
    print(f"{'=' * 70}")
    print(f"  Total articles:  {result['total_articles']}")
    print(f"  Event groups:    {result['total_groups']}")
    print(f"{'=' * 70}")

    for group in result["groups"]:
        print(f"\n  --- Group {group['group_id']}: \"{group['label']}\" ---")
        print(f"      Articles in group: {group['article_count']}")

        for article in group["articles"]:
            title = article.get("title", "N/A")
            source = article.get("source", {}).get("name", "N/A")
            is_rep = " [REPRESENTATIVE]" if article is group["representative"] else ""
            print(f"        - \"{title}\" ({source}){is_rep}")

    print()

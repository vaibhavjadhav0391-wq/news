"""
End-to-End Pipeline Orchestrator - Stage 6A
Connects all Stage 1-5 modules into a single analysis pipeline.

PIPELINE FLOW:
==============

  User topic/claim
       |
  1. FETCH  (news_fetcher.fetch_articles)
       |
  2. DEDUPLICATE  (deduplicator.find_duplicates)
       |
  3. GROUP  (grouper.group_articles)
       |
  4. COMPARE  (comparator.compare_all_groups)
       |
  5. ANALYZE  (gemini_analyzer.analyze_all_groups)
       |
  Final structured result

Each stage can still be used and tested independently.
This module simply wires them together and handles errors at each step.
"""

from news_fetcher import fetch_articles
from processing.relevance_filter import filter_relevant_articles, DEFAULT_RELEVANCE_THRESHOLD
from processing.deduplicator import find_duplicates, print_dedup_report
from processing.grouper import group_articles, print_group_report
from analysis.comparator import compare_all_groups, print_comparison_report
from analysis.gemini_analyzer import (
    analyze_all_groups,
    create_gemini_client,
    print_analysis_report,
)


# ──────────────────────────────────────────────────────────────
# Pipeline result helpers
# ──────────────────────────────────────────────────────────────

def _make_result(status, stage, topic, **kwargs):
    """Build a standardized pipeline result dict."""
    result = {
        "status": status,          # "success", "partial", or "error"
        "failed_stage": None if status == "success" else stage,
        "topic": topic,
        "articles_fetched": 0,
        "articles_filtered_out": 0,
        "articles_after_relevance_filter": 0,
        "duplicates_removed": 0,
        "unique_articles": 0,
        "event_groups": 0,
        "groups": [],
        "comparisons": [],
        "analyses": [],
        "error": None,
    }
    result.update(kwargs)
    return result


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(topic, page_size=30, dedup_threshold=0.65,
                 group_threshold=0.30, relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
                 gemini_client=None, skip_gemini=False,
                 _fetch_fn=None):
    """
    Run the complete news analysis pipeline.

    Args:
        topic: News topic or claim to search for.
        page_size: Number of articles to fetch (default: 30).
        dedup_threshold: Similarity threshold for deduplication (default: 0.65).
        group_threshold: Similarity threshold for event grouping (default: 0.30).
        relevance_threshold: Strict relevance filter threshold (default: 0.50).
        gemini_client: Optional pre-created Gemini client (for testing/reuse).
        skip_gemini: If True, skip the Gemini analysis step (stages 1-4 only).
        _fetch_fn: Internal override for the fetch function (for testing).

    Returns:
        Pipeline result dict with:
            "status": "success", "partial", or "error"
            "failed_stage": name of failing stage, or None on success
            "topic": the search topic
            "articles_fetched": count of raw articles
            "articles_filtered_out": count of articles rejected by relevance filter
            "articles_after_relevance_filter": count of articles passing relevance filter
            "duplicates_removed": count of duplicates removed
            "unique_articles": count of unique articles
            "event_groups": count of event groups
            "comparisons": list of comparison results
            "analyses": list of AI analysis results (empty if skip_gemini)
            "error": error message string, or None
    """
    # ── Validate input ──
    if not topic or not topic.strip():
        return _make_result("error", "input", topic or "",
                            error="No topic provided. Please enter a news topic or claim.")

    topic = topic.strip()
    fetcher = _fetch_fn or fetch_articles

    # ── Stage 1: Fetch articles ──
    try:
        articles = fetcher(topic, page_size=page_size)
    except Exception as e:
        return _make_result("error", "fetch", topic,
                            error=f"Failed to fetch articles: {e}")

    if not articles:
        return _make_result("error", "fetch", topic,
                            error=f"No articles found for \"{topic}\". Try a different topic.")

    articles_fetched = len(articles)

    # ── Stage 1B: Strict Relevance Filter ──
    try:
        rel_result = filter_relevant_articles(topic, articles, threshold=relevance_threshold)
        relevant_articles = rel_result.get("relevant_articles", [])
        articles_filtered_out = rel_result.get("stats", {}).get("rejected", 0)
        articles_after_relevance_filter = len(relevant_articles)
    except Exception as e:
        return _make_result("error", "relevance_filter", topic,
                            articles_fetched=articles_fetched,
                            error=f"Relevance filter failed: {e}")

    if not relevant_articles:
        return _make_result("error", "relevance_filter", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=0,
                            error=f"No articles relevant to the query were found for \"{topic}\". Try a different or broader topic.")

    # ── Stage 2: Deduplicate ──
    try:
        dedup_result = find_duplicates(relevant_articles, threshold=dedup_threshold)
        unique_articles = dedup_result.get("unique_articles", relevant_articles)
        # NOTE: find_duplicates() returns the removed-duplicate count under the
        # key "total_removed" (see processing/deduplicator.py), not
        # "duplicates_removed". Using the wrong key here silently defaulted
        # this statistic to 0 on every run, even when duplicates were found
        # and correctly stripped out of unique_articles.
        duplicates_removed = dedup_result.get("total_removed", 0)
    except Exception as e:
        return _make_result("error", "deduplication", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=articles_after_relevance_filter,
                            error=f"Deduplication failed: {e}")

    if not unique_articles:
        return _make_result("error", "deduplication", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=articles_after_relevance_filter,
                            duplicates_removed=duplicates_removed,
                            error="All articles were duplicates. No unique articles remain.")

    unique_count = len(unique_articles)

    # ── Stage 3: Group by event ──
    try:
        grouping_result = group_articles(unique_articles, threshold=group_threshold)
    except Exception as e:
        return _make_result("error", "grouping", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=articles_after_relevance_filter,
                            duplicates_removed=duplicates_removed,
                            unique_articles=unique_count,
                            error=f"Event grouping failed: {e}")

    groups = grouping_result.get("groups", [])
    if not groups:
        return _make_result("error", "grouping", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=articles_after_relevance_filter,
                            duplicates_removed=duplicates_removed,
                            unique_articles=unique_count,
                            error="No event groups formed from the articles.")

    event_group_count = len(groups)

    # ── Stage 4: Compare sources within each group ──
    try:
        comparisons = compare_all_groups(grouping_result)
    except Exception as e:
        return _make_result("error", "comparison", topic,
                            articles_fetched=articles_fetched,
                            articles_filtered_out=articles_filtered_out,
                            articles_after_relevance_filter=articles_after_relevance_filter,
                            duplicates_removed=duplicates_removed,
                            unique_articles=unique_count,
                            event_groups=event_group_count,
                            groups=groups,
                            error=f"Comparison failed: {e}")

    # ── Stage 5: Gemini analysis (optional) ──
    analyses = []
    if not skip_gemini:
        try:
            if gemini_client is None:
                gemini_client = create_gemini_client()
            analyses = analyze_all_groups(comparisons, client=gemini_client)
        except Exception as e:
            # Partial success: stages 1-4 completed, Gemini failed
            return _make_result("partial", "gemini", topic,
                                articles_fetched=articles_fetched,
                                articles_filtered_out=articles_filtered_out,
                                articles_after_relevance_filter=articles_after_relevance_filter,
                                duplicates_removed=duplicates_removed,
                                unique_articles=unique_count,
                                event_groups=event_group_count,
                                groups=groups,
                                comparisons=comparisons,
                                analyses=[],
                                error=f"Gemini analysis failed: {e}")

        # Check for individual group Gemini errors (partial failure)
        gemini_errors = [a for a in analyses if a.get("status") == "error"]
        if gemini_errors and len(gemini_errors) == len(analyses):
            return _make_result("partial", "gemini", topic,
                                articles_fetched=articles_fetched,
                                articles_filtered_out=articles_filtered_out,
                                articles_after_relevance_filter=articles_after_relevance_filter,
                                duplicates_removed=duplicates_removed,
                                unique_articles=unique_count,
                                event_groups=event_group_count,
                                groups=groups,
                                comparisons=comparisons,
                                analyses=analyses,
                                error="Gemini analysis failed for all groups.")

    return _make_result("success", None, topic,
                        articles_fetched=articles_fetched,
                        articles_filtered_out=articles_filtered_out,
                        articles_after_relevance_filter=articles_after_relevance_filter,
                        duplicates_removed=duplicates_removed,
                        unique_articles=unique_count,
                        event_groups=event_group_count,
                        groups=groups,
                        comparisons=comparisons,
                        analyses=analyses)


# ──────────────────────────────────────────────────────────────
# Report Printing
# ──────────────────────────────────────────────────────────────

def print_pipeline_report(result):
    """
    Print a human-readable summary of the full pipeline result.

    Args:
        result: Dict returned by run_pipeline().
    """
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE REPORT")
    print(f"{'=' * 70}")
    print(f"  Topic:              \"{result['topic']}\"")
    print(f"  Status:             {result['status'].upper()}")
    print(f"  Articles fetched:   {result['articles_fetched']}")
    print(f"  Duplicates removed: {result['duplicates_removed']}")
    print(f"  Unique articles:    {result['unique_articles']}")
    print(f"  Event groups:       {result['event_groups']}")
    print(f"  Comparisons:        {len(result['comparisons'])}")
    print(f"  AI analyses:        {len(result['analyses'])}")

    if result["error"]:
        print(f"\n  ERROR: {result['error']}")
        if result["failed_stage"]:
            print(f"  Failed at stage:    {result['failed_stage']}")

    # Print comparisons
    for comp in result["comparisons"]:
        print_comparison_report(comp)

    # Print AI analyses
    for analysis in result["analyses"]:
        print_analysis_report(analysis)

    print(f"\n{'=' * 70}")
    print(f"  END OF PIPELINE REPORT")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    """Interactive CLI entry point for the complete pipeline."""
    from config import validate_config, validate_gemini_config

    validate_config()
    validate_gemini_config()

    topic = input("\nEnter a news topic or claim: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    print(f'\nRunning full analysis pipeline for "{topic}"...\n')
    result = run_pipeline(topic)
    print_pipeline_report(result)


if __name__ == "__main__":
    main()

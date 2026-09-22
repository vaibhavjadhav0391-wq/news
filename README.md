# News Analyzer

A multi-source news comparison and analysis system. Given a topic or claim,
it retrieves candidate articles, strictly filters them for relevance,
removes duplicates, groups articles by real-world event, deterministically
compares what each source reports, and (optionally) uses Google's Gemini
model to produce a neutral, evidence-based synthesis — surfaced through a
small Flask web UI alongside the original source articles.

## Pipeline

```
User topic
    |
1. FETCH            news_fetcher.fetch_articles()      -> NewsAPI /v2/everything
    |
2. RELEVANCE FILTER  processing/relevance_filter.py     -> strict, deterministic, precision-first
    |
3. DEDUPLICATE       processing/deduplicator.py          -> TF-IDF cosine similarity
    |
4. GROUP BY EVENT    processing/grouper.py               -> single-linkage clustering
    |
5. COMPARE SOURCES   analysis/comparator.py              -> deterministic, neutral comparison
    |
6. AI SYNTHESIS      analysis/gemini_analyzer.py         -> Gemini (optional, can be skipped)
    |
Flask app (app.py) -> templates/index.html
```

Each stage is independently testable and the orchestrator
(`pipeline.run_pipeline`) wires them together, tracking statistics
(`articles_fetched`, `articles_filtered_out`, `articles_after_relevance_filter`,
`duplicates_removed`, `event_groups`, etc.) and stopping gracefully with a
clear status (`success`, `partial`, or `error`) at whichever stage fails.

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy your API keys into `.env` (already gitignored):
   ```
   NEWS_API_KEY=your_newsapi_key
   GEMINI_API_KEY=your_gemini_key
   ```
   - Get a free NewsAPI key: https://newsapi.org/register
   - Get a Gemini API key: https://aistudio.google.com/apikey

## Environment variables

| Variable         | Required for                        |
|-------------------|--------------------------------------|
| `NEWS_API_KEY`    | Fetching articles (Stage 1)          |
| `GEMINI_API_KEY`  | AI synthesis (Stage 6); the app runs fine without it — Stages 1-4 (comparison data + source evidence) still work, and the pipeline reports a `partial` result if Gemini fails or is unavailable. |

## Running the app

```
python app.py
```

Then open http://localhost:5000. The Flask development server is used for
local development (`debug=True`); it is not intended for production
deployment.

## Running the tests

The test files are standalone scripts (not pytest-based) — each prints
`PASS`/`FAIL` per assertion and exits non-zero on any failure:

```
python test_relevance_filter.py
python test_deduplicator.py
python test_grouper.py
python test_comparator.py
python test_gemini_analyzer.py
python test_pipeline.py
python test_app.py
```

All Gemini- and NewsAPI-dependent tests use `unittest.mock` — no real API
calls or network access are required to run the suite. For live,
non-mocked validation of the relevance filter against real NewsAPI
results, see `scripts/test_relevance_live.py` (requires a valid
`NEWS_API_KEY`; makes real network requests; never calls Gemini or prints
API keys).

## Design notes / known limitations

- The relevance filter (`DEFAULT_RELEVANCE_THRESHOLD = 0.50`) is
  intentionally precision-first: it will reject borderline articles rather
  than risk including off-topic results. Some legitimate queries may
  return "No relevant articles found" if NewsAPI's candidate pool for that
  topic is weak — this is filtering behaving as designed, not a bug.
- The comparator never labels a source as "fake" or "wrong" — differences
  between sources are reported neutrally; "potential contradictions" are
  flagged for the reader's own review, not treated as proven falsehoods.
- Gemini failures degrade gracefully: if Stages 1-4 succeed but Gemini
  fails, the app still shows the deterministic comparison data and source
  evidence rather than discarding the whole result.
- The Flask app runs with `debug=True`, which is appropriate for local
  development only.

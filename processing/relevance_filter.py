"""
Strict Article Relevance Filter - Stage 1B
Filters out news articles that are not genuinely relevant to the user query BEFORE
deduplication and downstream analysis.

Design Principles:
1. Precision over Recall: False negatives are preferable to false positives.
2. Deterministic & Fast: Uses string normalization, explicit controlled expansion,
   TF-IDF cosine similarity, and hard-reject entity rules. No external APIs or LLMs.
3. Title Priority: Article titles carry higher relevance weight than descriptions.
4. Entity Strictness: Missing critical anchor entities (countries, companies, products)
   triggers immediate rejection.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default threshold for precision-first filtering.
# Scores range from 0.0 to 1.0. Articles scoring below this threshold are rejected.
DEFAULT_RELEVANCE_THRESHOLD = 0.50

# Stop words to ignore during query concept extraction.
COMMON_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "who", "will", "with", "news", "latest",
    "today", "update", "updates", "report", "reports", "new", "top", "best"
}

# Explicit, controlled concept alias dictionary for domain expansion.
# Kept deliberately limited to avoid loose, off-topic matching.
ALIAS_DICTIONARY = {
    # Tech & AI
    "technology": {"technology", "tech", "technological", "space", "lunar", "spacecraft", "satellite", "isro", "nasa", "processor", "chip", "macbook", "computer", "computing", "device", "software", "hardware", "ai", "artificial intelligence"},
    "tech": {"technology", "tech", "technological", "space", "lunar", "spacecraft", "satellite", "isro", "nasa", "processor", "chip", "macbook", "computer", "computing", "device", "software", "hardware", "ai", "artificial intelligence"},
    "ai": {"ai", "artificial intelligence", "machine learning", "generative ai", "foundation model", "llm", "llms"},
    "artificial": {"artificial intelligence", "ai"},
    "intelligence": {"artificial intelligence", "ai"},
    "model": {"model", "models", "llm", "llms"},
    "models": {"model", "models", "llm", "llms"},
    # Governance & Legal
    "regulations": {"regulation", "regulations", "rule", "rules", "law", "laws", "legislation", "policy", "framework", "guidelines", "bill"},
    "regulation": {"regulation", "regulations", "rule", "rules", "law", "laws", "legislation", "policy", "framework", "guidelines", "bill"},
    "rules": {"rule", "rules", "regulation", "regulations", "legislation", "policy", "framework", "guidelines", "law", "laws"},
    "rule": {"rule", "rules", "regulation", "regulations", "legislation", "policy", "framework", "guidelines", "law", "laws"},
    "policy": {"policy", "policies", "framework", "regulation", "regulations", "rules"},
    "climate": {"climate", "environment", "environmental", "global warming", "emissions"},
    # Entity & Geography
    "india": {"india", "indian", "government of india", "indian government", "bharat", "delhi", "new delhi"},
    "indian": {"india", "indian", "government of india", "indian government", "bharat", "delhi", "new delhi"},
    "eu": {"eu", "europe", "european union", "european parliament", "brussels"},
    "europe": {"eu", "europe", "european union", "european parliament"},
    "european": {"eu", "europe", "european union", "european parliament"},
    "us": {"us", "usa", "united states", "america", "american", "washington"},
    "usa": {"us", "usa", "united states", "america", "american", "washington"},
    "uk": {"uk", "united kingdom", "britain", "british", "london"},
    "china": {"china", "chinese", "beijing"},
    # Tech Companies & Products
    "openai": {"openai", "open ai", "chatgpt"},
    "google": {"google", "alphabet", "gemini", "bard"},
    "apple": {"apple", "cupertino"},
    "samsung": {"samsung"},
    "iphone": {"iphone", "ios"},
    "galaxy": {"galaxy"},
    "rbi": {"rbi", "reserve bank of india"},
    "isro": {"isro", "indian space research organisation", "indian space research organization"},
}

# Known anchor entities (locations, companies, organizations, products).
# Queries containing these require evidence of the entity in the article text.
KNOWN_ENTITIES = {
    "india", "indian", "eu", "europe", "european", "us", "usa", "uk", "china",
    "openai", "google", "apple", "samsung", "iphone", "galaxy", "rbi", "isro",
    "microsoft", "meta", "amazon", "nvidia", "tesla"
}


def normalize_text(text: str) -> str:
    """
    Clean and normalize text for query/article matching.

    Steps:
        1. Convert to string and lowercase.
        2. Replace hyphens with spaces to keep compound words separate (e.g. "multi-source" -> "multi source").
        3. Remove punctuation except alphanumeric and spaces.
        4. Collapse multiple spaces into single space and strip whitespace.

    Args:
        text: Raw input string or None.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    text = str(text).lower()

    # Replace hyphens/slashes with spaces for better tokenization
    text = re.sub(r"[-/]", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_concepts(query: str) -> dict:
    """
    Extract structured concepts, anchor entities, and key phrases from a user query.

    Args:
        query: Raw search query string.

    Returns:
        dict containing:
            "raw_query": str,
            "normalized_query": str,
            "tokens": list of core word tokens,
            "concepts": list of dicts {"term": str, "aliases": set, "is_entity": bool},
            "anchor_entities": list of concept dicts for required entities,
            "query_phrases": list of multi-word phrase strings
    """
    raw_query = query if query is not None else ""
    norm_query = normalize_text(raw_query)

    if not norm_query:
        return {
            "raw_query": raw_query,
            "normalized_query": "",
            "tokens": [],
            "concepts": [],
            "anchor_entities": [],
            "query_phrases": [],
        }

    # Split into raw tokens
    words = norm_query.split()

    # Filter out stopwords to find core tokens
    tokens = [w for w in words if w not in COMMON_STOPWORDS and len(w) > 1]

    # If all words were stopwords, fallback to all tokens
    if not tokens and words:
        tokens = words

    # Detect anchor entities from raw query casing (capitalized words) or KNOWN_ENTITIES list
    raw_words = raw_query.split()
    capitalized_entities = set()
    for w in raw_words:
        cleaned_w = w.strip(string.punctuation).lower()
        if cleaned_w and cleaned_w not in COMMON_STOPWORDS:
            # Check if original had capital letter or is known entity
            if w[0].isupper() or cleaned_w in KNOWN_ENTITIES:
                capitalized_entities.add(cleaned_w)

    concepts = []
    anchor_entities = []
    seen_primary = set()

    for token in tokens:
        if token in seen_primary:
            continue
        seen_primary.add(token)

        aliases = ALIAS_DICTIONARY.get(token, {token}) | {token}
        is_entity = token in KNOWN_ENTITIES or token in capitalized_entities

        concept_dict = {
            "term": token,
            "aliases": aliases,
            "is_entity": is_entity,
        }
        concepts.append(concept_dict)

        if is_entity:
            anchor_entities.append(concept_dict)

    # Extract 2-word and 3-word query phrases for exact phrase matching
    query_phrases = []
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            query_phrases.append(f"{tokens[i]} {tokens[i+1]}")
    if len(tokens) >= 3:
        for i in range(len(tokens) - 2):
            query_phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    return {
        "raw_query": raw_query,
        "normalized_query": norm_query,
        "tokens": tokens,
        "concepts": concepts,
        "anchor_entities": anchor_entities,
        "query_phrases": query_phrases,
    }


def calculate_relevance_score(query: str, article: dict, threshold: float = DEFAULT_RELEVANCE_THRESHOLD) -> dict:
    """
    Calculate deterministic relevance score (0.0 to 1.0) and decision for a single article.

    Scoring formula:
        Composite Score = 0.40 * S_title + 0.20 * S_desc + 0.15 * S_phrase + 0.25 * S_tfidf

    Hard Reject Triggers:
        1. Malformed/missing article dict or missing title + description -> Reject (0.0)
        2. Anchor entity in query is missing from both title and description -> Reject (0.0)
        3. Multi-concept query (>=2 concepts) has ZERO concept overlap in title -> Reject (0.0)

    Args:
        query: User search query string.
        article: Article dictionary (NewsAPI format).
        threshold: Minimum score required to pass.

    Returns:
        dict with:
            "score": float (0.0 to 1.0),
            "passed": bool,
            "rejection_reason": str or None,
            "title_score": float,
            "desc_score": float,
            "phrase_score": float,
            "tfidf_score": float
    """
    if not isinstance(article, dict):
        return {
            "score": 0.0,
            "passed": False,
            "rejection_reason": "Malformed article object (not a dictionary)",
            "title_score": 0.0,
            "desc_score": 0.0,
            "phrase_score": 0.0,
            "tfidf_score": 0.0,
        }

    raw_title = article.get("title")
    raw_desc = article.get("description")

    title_norm = normalize_text(raw_title)
    desc_norm = normalize_text(raw_desc)

    # Hard Reject Rule: Missing both title and description
    if not title_norm and not desc_norm:
        return {
            "score": 0.0,
            "passed": False,
            "rejection_reason": "Missing both title and description",
            "title_score": 0.0,
            "desc_score": 0.0,
            "phrase_score": 0.0,
            "tfidf_score": 0.0,
        }

    parsed_query = extract_query_concepts(query)
    concepts = parsed_query["concepts"]
    anchor_entities = parsed_query["anchor_entities"]

    if not concepts:
        # If query is empty or whitespace only
        return {
            "score": 0.0,
            "passed": False,
            "rejection_reason": "Empty search query",
            "title_score": 0.0,
            "desc_score": 0.0,
            "phrase_score": 0.0,
            "tfidf_score": 0.0,
        }

    combined_text = f"{title_norm} {desc_norm}".strip()

    # ── HARD REJECT RULE 1: Missing Critical Anchor Entity ──
    for entity in anchor_entities:
        entity_found = any(alias in combined_text for alias in entity["aliases"])
        if not entity_found:
            return {
                "score": 0.0,
                "passed": False,
                "rejection_reason": f"Missing critical entity: '{entity['term']}'",
                "title_score": 0.0,
                "desc_score": 0.0,
                "phrase_score": 0.0,
                "tfidf_score": 0.0,
            }

    # ── HARD REJECT RULE 2: Zero Title Concept Overlap for Multi-Concept Query ──
    # If title exists and there are 2+ query concepts, title must contain at least 1 query concept
    if title_norm and len(concepts) >= 2:
        title_has_concept = False
        for c in concepts:
            if any(alias in title_norm for alias in c["aliases"]):
                title_has_concept = True
                break

        if not title_has_concept:
            return {
                "score": 0.0,
                "passed": False,
                "rejection_reason": "Title contains zero query concepts for multi-concept query",
                "title_score": 0.0,
                "desc_score": 0.0,
                "phrase_score": 0.0,
                "tfidf_score": 0.0,
            }

    # ── SCORE CALCULATIONS ──

    # 1. Title Concept Score (S_title)
    if title_norm:
        matched_title_concepts = sum(
            1 for c in concepts if any(alias in title_norm for alias in c["aliases"])
        )
        s_title = matched_title_concepts / len(concepts)
    else:
        s_title = 0.0

    # 2. Description Concept Score (S_desc)
    if desc_norm:
        matched_desc_concepts = sum(
            1 for c in concepts if any(alias in desc_norm for alias in c["aliases"])
        )
        s_desc = matched_desc_concepts / len(concepts)
    else:
        s_desc = 0.0

    # 3. Exact Phrase Score (S_phrase)
    s_phrase = 0.0
    query_phrases = parsed_query["query_phrases"]
    if query_phrases:
        phrase_matches = 0
        for phrase in query_phrases:
            if phrase in title_norm:
                phrase_matches += 1.0
            elif phrase in desc_norm:
                phrase_matches += 0.5
        s_phrase = min(1.0, phrase_matches / len(query_phrases))
    elif len(concepts) == 1:
        # Single concept query gets phrase bonus if term matches title exactly
        term = concepts[0]["term"]
        if term in title_norm:
            s_phrase = 1.0
        elif term in desc_norm:
            s_phrase = 0.5

    # 4. TF-IDF Cosine Similarity Score (S_tfidf)
    s_tfidf = 0.0
    try:
        # Weight title more heavily by repeating it
        weighted_article_text = f"{title_norm} {title_norm} {desc_norm}".strip()
        corpus = [parsed_query["normalized_query"], weighted_article_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        s_tfidf = max(0.0, min(1.0, float(sim)))
    except Exception:
        s_tfidf = 0.0

    # ── COMPOSITE SCORE ──
    # Conceptual Weighting: Title Concept (40%), Desc Concept (20%), Phrase (15%), TF-IDF (25%)
    composite_score = (0.40 * s_title) + (0.20 * s_desc) + (0.15 * s_phrase) + (0.25 * s_tfidf)
    final_score = round(min(1.0, max(0.0, composite_score)), 4)

    passed = final_score >= threshold
    reason = None if passed else f"Relevance score {final_score:.2f} is below threshold {threshold}"

    return {
        "score": final_score,
        "passed": passed,
        "rejection_reason": reason,
        "title_score": round(s_title, 4),
        "desc_score": round(s_desc, 4),
        "phrase_score": round(s_phrase, 4),
        "tfidf_score": round(s_tfidf, 4),
    }


def filter_relevant_articles(
    query: str,
    articles: list[dict],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    return_rejected: bool = True
) -> dict:
    """
    Filter a list of articles for strict query relevance.

    Args:
        query: User search query string.
        articles: List of NewsAPI article dictionaries.
        threshold: Minimum score required to pass (0.0 to 1.0). Default: 0.50.
        return_rejected: If True, includes rejected articles with scores and reasons.

    Returns:
        dict with:
            "relevant_articles": list of accepted article dicts (original dicts unchanged),
            "rejected_articles": list of rejected article dicts (with score metadata if enabled),
            "stats": {
                "total_articles": int,
                "accepted": int,
                "rejected": int,
                "threshold": float,
            }
    """
    if not articles:
        return {
            "relevant_articles": [],
            "rejected_articles": [],
            "stats": {
                "total_articles": 0,
                "accepted": 0,
                "rejected": 0,
                "threshold": threshold,
            },
        }

    relevant_articles = []
    rejected_articles = []

    for article in articles:
        eval_res = calculate_relevance_score(query, article, threshold=threshold)

        if eval_res["passed"]:
            relevant_articles.append(article)
        else:
            if return_rejected:
                # Add score metadata without modifying original article keys
                rejected_info = dict(article) if isinstance(article, dict) else {}
                rejected_info["relevance_score"] = eval_res["score"]
                rejected_info["rejection_reason"] = eval_res["rejection_reason"]
                rejected_articles.append(rejected_info)
            else:
                rejected_articles.append(article)

    return {
        "relevant_articles": relevant_articles,
        "rejected_articles": rejected_articles,
        "stats": {
            "total_articles": len(articles),
            "accepted": len(relevant_articles),
            "rejected": len(rejected_articles),
            "threshold": threshold,
        },
    }

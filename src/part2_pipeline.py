"""
part2_pipeline.py — Multi-source RAG pipeline with query routing.

Architecture:
    User Query → classify_query() → retrieve_context() → generate_answer()

The router sends queries to:
    - CSV only   : sales analytics questions
    - Text only  : product features / review questions
    - Both       : questions that need to combine both sources
"""

import glob
import json
import os
import re

import pandas as pd

from src.config import CSV_PATH, GROQ_MODEL, MAX_CONTEXT_CHARS, TEXT_DIR, get_client

# ── Module-level client (initialized lazily on first use) ─────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def _get_txt_files() -> list[str]:
    """Return sorted list of all .txt product page files."""
    return sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))


# ── Module 1: Query router ─────────────────────────────────────────────────────

def classify_query(question: str) -> dict:
    """
    Use the LLM to classify the question and select the appropriate data source(s).

    Args:
        question: The user's natural language question.

    Returns:
        A dict with keys:
          - route     : 'csv' | 'text' | 'both'
          - reasoning : one-sentence explanation
          - csv_hint  : what to filter/aggregate from CSV (or None)
          - text_hint : what keywords/files to search in text (or None)
    """
    client = _get_client()
    txt_files = _get_txt_files()
    available_files = [os.path.basename(f) for f in txt_files]

    prompt = f"""You are a query router for a multi-source RAG system with two data sources:

1. CSV (structured): daily_sales.csv
   - Columns: date, product_id, product_name, category, units_sold, unit_price, total_revenue, region
   - 1000 rows, Oct–Dec 2024, 35 products, regions: North/South/East/West/Central
   - Use for: revenue totals, sales volume, regional comparisons, date filtering

2. Text (unstructured): product page .txt files
   - Available files: {available_files}
   - Content: product descriptions, specifications, customer reviews
   - Use for: product features, review sentiment, specifications, recommendations

Routing rules:
- Route to 'csv'  when the question is purely about numbers, sales, revenue, or regions
- Route to 'text' when the question is purely about product features, specs, or customer reviews
- Route to 'both' when the question needs to combine sales performance with product quality/reviews

User question: {question}

Return ONLY valid JSON (no markdown, no extra text):
{{
    "route": "csv | text | both",
    "reasoning": "one sentence explaining the routing decision",
    "csv_hint": "what columns/filters/aggregations to apply (or null)",
    "text_hint": "what keywords or product files to search (or null)"
}}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Warning] Router JSON parse failed. Raw output:\n{raw}")
        return {
            "route": "both",
            "reasoning": "Parse failed, defaulting to both sources",
            "csv_hint": "general sales data",
            "text_hint": "product details",
        }


# ── Module 2: CSV retriever ────────────────────────────────────────────────────

def retrieve_from_csv(question: str, csv_hint: str) -> str:
    """
    Retrieve relevant context from the sales CSV using pandas.

    Generates a static summary (category revenue, region volume, top products)
    and applies dynamic filters based on keywords found in csv_hint.

    Args:
        question: The user's question.
        csv_hint: Guidance from the router on what to filter/aggregate.

    Returns:
        A formatted string with CSV statistics relevant to the question.
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    hint_lower = (csv_hint or "").lower()
    summary_parts = []

    # Static overview (always included)
    summary_parts.append("=== CSV Overview ===")
    summary_parts.append(
        f"Total rows: {len(df)}, "
        f"Date range: {df['date'].min().date()} to {df['date'].max().date()}"
    )
    summary_parts.append(f"Categories: {sorted(df['category'].unique())}")
    summary_parts.append(f"Regions: {sorted(df['region'].unique())}")

    # Revenue by category
    cat_rev = (
        df.groupby("category")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    summary_parts.append("\n=== Total Revenue by Category ===")
    summary_parts.append(cat_rev.to_string())

    # Sales volume by region
    region_vol = (
        df.groupby("region")["units_sold"]
        .sum()
        .sort_values(ascending=False)
    )
    summary_parts.append("\n=== Total Units Sold by Region ===")
    summary_parts.append(region_vol.to_string())

    # Dynamic: filter by month if mentioned in hint
    for month_name, month_num in [("october", 10), ("november", 11), ("december", 12)]:
        if month_name in hint_lower or month_name[:3] in hint_lower:
            monthly = df[df["date"].dt.month == month_num]
            monthly_rev = (
                monthly.groupby("category")["total_revenue"]
                .sum()
                .sort_values(ascending=False)
            )
            summary_parts.append(f"\n=== Revenue by Category in {month_name.capitalize()} ===")
            summary_parts.append(monthly_rev.to_string())

    # Dynamic: filter by category if mentioned
    for category in df["category"].unique():
        if category.lower() in hint_lower:
            cat_df = df[df["category"] == category]
            cat_monthly = cat_df.groupby(cat_df["date"].dt.month)["total_revenue"].sum()
            summary_parts.append(f"\n=== {category} Monthly Revenue ===")
            summary_parts.append(cat_monthly.to_string())

    # Dynamic: filter by region if mentioned
    for region in df["region"].unique():
        if region.lower() in hint_lower:
            region_df = df[df["region"] == region]
            region_products = (
                region_df.groupby("product_name")["units_sold"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            summary_parts.append(f"\n=== Top Products in {region} Region ===")
            summary_parts.append(region_products.to_string())

    # Top 15 products overall
    top_products = (
        df.groupby("product_name")
        .agg(total_revenue=("total_revenue", "sum"), total_units=("units_sold", "sum"))
        .sort_values("total_revenue", ascending=False)
        .head(15)
    )
    summary_parts.append("\n=== Top 15 Products by Revenue ===")
    summary_parts.append(top_products.to_string())

    context = "\n".join(summary_parts)

    # Truncate if too long
    limit = MAX_CONTEXT_CHARS // 2
    if len(context) > limit:
        context = context[:limit] + "\n... [truncated]"

    return context


# ── Module 3: Text retriever ───────────────────────────────────────────────────

def retrieve_from_text(question: str, text_hint: str) -> str:
    """
    Retrieve relevant content from product page text files using keyword scoring.

    Args:
        question:  The user's question.
        text_hint: Keywords or product names from the router to guide search.

    Returns:
        A string with the most relevant product page content.
    """
    txt_files = _get_txt_files()
    hint_lower = (text_hint or "").lower()

    # Noise words that appear in every product page and don't discriminate between
    # files.  We strip them from both the question and hint before building score_words.
    _noise = {
        # routing instruction words
        "search", "find", "look", "file", "files", "keyword", "keywords",
        "page", "pages", "using", "also", "then", "them", "like",
        # generic English function / linking words
        "about", "what", "that", "this", "with", "from", "their", "there",
        "these", "those", "have", "been", "will", "which", "well", "want",
        # e-commerce boilerplate present in every product page
        "product", "products", "customer", "customers", "review", "reviews",
        "rating", "ratings", "highly", "rated", "recommend", "high",
        "quality", "brand", "category", "price", "determine",
        # geographic / sales query terms (not product content)
        "region", "sells", "west", "east", "north", "south", "central",
    }
    question_clean = re.sub(r"[^\w\s]", " ", question.lower())
    hint_clean = re.sub(r"[^\w\s]", " ", hint_lower)

    q_words = [w for w in question_clean.split() if len(w) > 3 and w not in _noise]
    h_words = [
        w for w in hint_clean.split()
        if len(w) > 3
        and w not in _noise
        and not w.endswith("_product_page")  # drop bare SKU filename tokens
    ]
    score_words = list(set(q_words + h_words))

    # Read all files once
    file_data = []
    for filepath in txt_files:
        with open(filepath, "r") as f:
            content = f.read()
        file_data.append((filepath, content))

    n_docs = len(file_data)

    # IDF weighting: log(1 + N/(df+1))
    # Rare words (1-2 files) → weight ~1.6-1.8  (true product discriminators)
    # Universal words (all 10 files) → weight ~0.06  (still generic even after noise filter)
    from math import log as _log
    idf = {}
    for w in score_words:
        df = sum(1 for _, c in file_data if w in c.lower())
        idf[w] = _log(1 + n_docs / (df + 1))

    # Score each file: sum of IDF values for matching words
    file_scores = []
    for filepath, content in file_data:
        content_lower = content.lower()
        score = sum(idf[w] for w in score_words if w in content_lower)
        file_scores.append((score, filepath, content))

    file_scores.sort(key=lambda x: x[0], reverse=True)

    # Determine if there are STRONG discriminating words (IDF ≥ 1.0 in at least one file).
    max_matching_idf = max(
        (idf[w] for w in score_words if any(w in c.lower() for _, c in file_data)),
        default=0.0,
    )

    char_limit = MAX_CONTEXT_CHARS // 2

    def _front_back(text: str, n: int) -> str:
        """Return first n//2 chars + last n//2 chars of text (captures description + reviews)."""
        if len(text) <= n:
            return text
        half = n // 2
        return text[:half] + text[-half:]

    if max_matching_idf >= 1.0:
        # Specific query (Q3/Q4/Q6): rank by discriminating keywords, top-5 files.
        top_files = [t for t in file_scores if t[0] > 0.1][:5] or file_scores[:3]
        per_file = char_limit // len(top_files)
        context_parts = [
            f"=== File: {os.path.basename(fp)} (score: {s:.2f}) ===\n{_front_back(c, per_file)}"
            for s, fp, c in top_files
        ]
    else:
        # Comparative query (Q5: "best reviews"): no clear discriminator — show all
        # files with front+back sampling so LLM can compare ratings across products.
        per_file = char_limit // n_docs
        context_parts = [
            f"=== File: {os.path.basename(fp)} (score: {s:.2f}) ===\n{_front_back(c, per_file)}"
            for s, fp, c in file_scores
        ]

    return "\n\n".join(context_parts)


# ── Module 4: Context combiner ─────────────────────────────────────────────────

def retrieve_context(question: str, classification: dict) -> str:
    """
    Route the question to the appropriate retriever(s) and combine results.

    Args:
        question:       The user's question.
        classification: Output from classify_query().

    Returns:
        Combined context string ready to pass to the LLM.
    """
    route     = classification.get("route", "both")
    csv_hint  = classification.get("csv_hint", "") or ""
    text_hint = classification.get("text_hint", "") or ""

    print(f"[Router] Route     : {route.upper()}")
    print(f"[Router] Reasoning : {classification.get('reasoning')}")
    print(f"[Router] CSV hint  : {csv_hint}")
    print(f"[Router] Text hint : {text_hint}")

    context_parts = []

    if route in ("csv", "both"):
        print("\n[Retrieving] → CSV source...")
        csv_context = retrieve_from_csv(question, csv_hint)
        context_parts.append("### SOURCE: Structured Sales Data (CSV)\n" + csv_context)

    if route in ("text", "both"):
        print("[Retrieving] → Text source...")
        text_context = retrieve_from_text(question, text_hint)
        context_parts.append("### SOURCE: Unstructured Product Pages (Text)\n" + text_context)

    return "\n\n" + "\n\n".join(context_parts)


# ── Module 5: Answer generator ─────────────────────────────────────────────────

def generate_answer(question: str, context: str, route: str) -> str:
    """
    Generate a final answer using the LLM, grounded in the retrieved context.

    Args:
        question: The user's original question.
        context:  Combined context from retrieve_context().
        route:    'csv', 'text', or 'both' — used to tailor the system prompt.

    Returns:
        A detailed answer string.
    """
    client = _get_client()

    source_instruction = {
        "csv":  "Base your answer on the sales data. Include specific numbers, totals, and comparisons.",
        "text": "Base your answer on the product descriptions and customer reviews. Quote relevant review excerpts.",
        "both": "Combine insights from both sales data and product reviews. Cross-reference sales performance with product quality.",
    }.get(route, "Use all available context.")

    prompt = f"""You are an e-commerce analytics assistant. Answer the user's question using ONLY
the retrieved context below.

Instructions:
- {source_instruction}
- Be specific and cite exact figures or file names where relevant
- If context is insufficient, state what is missing
- Structure your answer clearly

=== Retrieved Context ===
{context}
=== End of Context ===

User question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )

    return response.choices[0].message.content.strip()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Full multi-source RAG pipeline: Classify → Retrieve → Generate.

    Args:
        question: A natural language question about sales data or products.

    Returns:
        A detailed, grounded answer string.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    # Step 1: Classify → decide which source(s) to use
    classification = classify_query(question)

    # Step 2: Retrieve → collect context from the selected source(s)
    print()
    context = retrieve_context(question, classification)

    # Step 3: Generate → produce the final answer
    print("\n[Generating answer...]")
    answer = generate_answer(question, context, classification.get("route", "both"))

    return answer

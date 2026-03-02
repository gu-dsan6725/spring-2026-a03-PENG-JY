"""
part1_pipeline.py — Code Q&A pipeline using bash tools.

Architecture:
    User Query → classify_query() → retrieve_context() → generate_answer()

The pipeline retrieves codebase context by executing bash commands (grep, find,
cat, tree) instead of vector search, then passes the output to an LLM.
"""

import json
import subprocess

from src.config import GROQ_MODEL, MAX_CONTEXT_CHARS, REPO_PATH, get_client


# ── Module-level client (initialized lazily on first use) ─────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


# ── Module 1: Bash tool executor ───────────────────────────────────────────────

def execute_bash(command: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Safely execute a bash command and return its stdout as a string.

    Args:
        command:   The bash command string to execute.
        max_chars: Maximum characters to return (truncates if exceeded).

    Returns:
        Command output as a string, or an error/timeout message.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout

        # Fall back to stderr if stdout is empty
        if result.stderr and not output:
            output = result.stderr

        # Truncate to avoid exceeding LLM context window
        if len(output) > max_chars:
            output = (
                output[:max_chars]
                + f"\n\n... [truncated — total {len(result.stdout)} chars]"
            )

        return output.strip() if output.strip() else "(no output)"

    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)"
    except Exception as e:
        return f"(execution error: {e})"


# ── Module 2: Query router ─────────────────────────────────────────────────────

def classify_query(question: str) -> dict:
    """
    Use the LLM to classify the question and generate bash commands for retrieval.

    Args:
        question: The user's natural language question about the codebase.

    Returns:
        A dict with keys: query_type, reasoning, commands (list of bash strings).
    """
    client = _get_client()

    prompt = f"""You are a codebase query router. Given a user question, decide which bash commands
to run to retrieve the most relevant information from the repository.

Repository path: {REPO_PATH}

Available tools:
- tree: show directory structure
- find: locate files by name or extension
- cat: read file contents
- grep: search for patterns inside files

Query type hints:
- dependency     → cat pyproject.toml, cat package.json
- structure      → tree -L 3, find with extension filters
- code_search    → grep for keywords + cat relevant files
- documentation  → find docs/ + cat markdown files
- multi          → combine grep, find, cat across code and docs

User question: {question}

Return ONLY valid JSON (no markdown, no extra text):
{{
    "query_type": "dependency | structure | code_search | documentation | multi",
    "reasoning": "one sentence explaining the chosen strategy",
    "commands": [
        "bash command 1",
        "bash command 2"
    ]
}}

Rules:
- Maximum 5 commands
- Each command must be directly executable in a terminal
- Use --include filters with grep to target specific file types
- For complex questions, first grep to find relevant files, then cat those files
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the LLM wraps output in ```json ... ```
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Warning] Router JSON parse failed, using fallback.\nRaw: {raw}")
        return {
            "query_type": "multi",
            "reasoning": "JSON parse failed, using fallback",
            "commands": [
                f"tree {REPO_PATH} -L 3 2>/dev/null",
                f"grep -r '{question[:40]}' {REPO_PATH} --include='*.py' --include='*.md' -l 2>/dev/null | head -10",
            ],
        }


# ── Module 3: Context retriever ────────────────────────────────────────────────

def retrieve_context(classification: dict) -> str:
    """
    Execute all bash commands from the router and collect their outputs.

    Args:
        classification: The dict returned by classify_query(), containing 'commands'.

    Returns:
        A single string with all command outputs concatenated, labeled by command.
    """
    context_parts = []
    commands = classification.get("commands", [])

    print(f"[Router] Query type : {classification.get('query_type')}")
    print(f"[Router] Reasoning  : {classification.get('reasoning')}")
    print(f"[Router] Executing {len(commands)} command(s):")

    per_command_limit = MAX_CONTEXT_CHARS // max(len(commands), 1)

    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
        output = execute_bash(cmd, max_chars=per_command_limit)
        context_parts.append(f"=== Command: {cmd} ===\n{output}")

    return "\n\n".join(context_parts)


# ── Module 4: Answer generator ─────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Generate a final answer using the LLM, grounded in the retrieved context.

    Args:
        question: The original user question.
        context:  The concatenated bash command outputs from retrieve_context().

    Returns:
        A detailed answer string with references to specific files.
    """
    client = _get_client()

    prompt = f"""You are an expert code assistant. Answer the user's question using ONLY
the context retrieved from the codebase below.

Requirements:
- Be accurate and specific
- Reference concrete file names and paths
- Quote relevant code snippets where helpful
- If the context is insufficient to fully answer, state which parts are uncertain

=== Retrieved Codebase Context ===
{context}
=== End of Context ===

User question: {question}

Provide a detailed answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    return response.choices[0].message.content.strip()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Full RAG pipeline: Classify → Retrieve → Generate.

    Args:
        question: A natural language question about the mcp-gateway-registry codebase.

    Returns:
        A detailed answer string grounded in actual codebase content.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    # Step 1: Classify the question → decide which bash commands to run
    classification = classify_query(question)

    # Step 2: Execute bash commands → collect codebase context
    print("\n[Retrieving context...]")
    context = retrieve_context(classification)

    # Step 3: Pass context + question to LLM → generate final answer
    print("\n[Generating answer...]")
    answer = generate_answer(question, context)

    return answer

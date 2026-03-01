"""
part1_pipeline.py — Code Q&A pipeline using bash tools.

Architecture:
    User Query → classify_query() → retrieve_context() → generate_answer()

The pipeline retrieves codebase context by executing bash commands (grep, find,
cat, tree) instead of vector search, then passes the output to an LLM.
"""

import json
import os
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

def _sanitize_commands(commands: list) -> list:
    """Replace disallowed or broken commands with safe equivalents."""
    sanitized = []
    for cmd in commands:
        # tree is not installed — replace with find
        if "tree " in cmd:
            cmd = (
                f"find {REPO_PATH} -maxdepth 3 -type f | sort | head -60"
            )
        sanitized.append(cmd)
    return sanitized


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

CRITICAL RULES:
- NEVER use the `tree` command — it is not installed. Use `find` instead.
- Return ONLY valid JSON with no markdown fences, no extra text.
- Maximum 5 commands, each directly executable in a terminal.

Available tools:
- find: locate files by name, extension, or depth
- cat: read file contents
- grep: search for patterns inside files (always use --include to filter file types)

=== FEW-SHOT EXAMPLES ===

Example 1 — entry_point question:
Q: "What is the main entry point file for the registry service?"
A:
{{
    "query_type": "entry_point",
    "reasoning": "Search for uvicorn calls and FastAPI app init to locate the registry service entry point.",
    "commands": [
        "grep -r 'uvicorn' {REPO_PATH} --include='*.py' --include='*.sh' -l",
        "find {REPO_PATH}/registry -name 'main.py'",
        "cat {REPO_PATH}/registry/main.py | head -60"
    ]
}}

Example 2 — structure/file-type question:
Q: "What programming languages and file types are used in this repository?"
A:
{{
    "query_type": "structure",
    "reasoning": "Enumerate file extensions across the whole repo to identify all languages and file types.",
    "commands": [
        "find {REPO_PATH} -type f | sed 's/.*\\.//' | sort | uniq -c | sort -rn | head -20",
        "find {REPO_PATH} -maxdepth 3 -type f | sort | head -60"
    ]
}}

Example 3 — dependency question:
Q: "What Python dependencies does this project use?"
A:
{{
    "query_type": "dependency",
    "reasoning": "Read pyproject.toml and any package.json to find declared dependencies.",
    "commands": [
        "cat {REPO_PATH}/pyproject.toml",
        "cat {REPO_PATH}/package.json 2>/dev/null || echo 'no package.json'"
    ]
}}

=== END EXAMPLES ===

Additional hints by query type:
- entry_point : registry service starts with uvicorn in registry/main.py — NOT credentials-provider/ OAuth scripts
- structure   : use find + sed to count extensions; never use tree
- code_search : grep for keywords then cat the relevant files
- documentation: find {REPO_PATH}/docs -name "*.md" | xargs grep -l keyword | head -5
- multi       : combine grep, find, cat across code and docs

User question: {question}

Return ONLY valid JSON (no markdown, no extra text):
{{
    "query_type": "dependency | structure | entry_point | code_search | documentation | multi",
    "reasoning": "one sentence explaining the chosen strategy",
    "commands": [
        "bash command 1",
        "bash command 2"
    ]
}}
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
        result = json.loads(raw)
        result["commands"] = _sanitize_commands(result.get("commands", []))
        return result
    except json.JSONDecodeError:
        print(f"[Warning] Router JSON parse failed, using fallback.\nRaw: {raw}")
        return {
            "query_type": "multi",
            "reasoning": "JSON parse failed, using fallback",
            "commands": _sanitize_commands([
                f"find {REPO_PATH} -maxdepth 3 -type f | sort | head -60",
                f"grep -rl '{question[:40]}' {REPO_PATH} --include='*.py' --include='*.md' 2>/dev/null | head -10",
            ]),
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

    # Warn early if the repo path doesn't exist — commands will silently return nothing
    if not os.path.isdir(REPO_PATH):
        warning = (
            f"[WARNING] Repository path '{REPO_PATH}' does not exist on disk. "
            "All bash commands will likely return empty output. "
            "Please clone the repository first."
        )
        print(warning)
        context_parts.append(warning)

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

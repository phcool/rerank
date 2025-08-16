#!/usr/bin/env python3
"""
Centralized ranking prompt builders to reduce duplication across scripts.
"""
from typing import Optional


def build_elimination_sort_instruction(query_text: Optional[str]) -> str:
    """
    Multi-round comparative ranking instruction:
    - In <think>, perform multiple rounds of pairwise comparisons between passages as needed
    - Do NOT output any full ranking in <think>
    - In <answer>, output ONLY the final ranking
    """
    q = f'"{query_text}"' if query_text else "the search query"
    return (
        f"Please rank these passages according to their relevance to the search query: {q}\n\n"
        "CRITICAL:\n"
        "- Do ALL reasoning inside <think> ... </think>\n"
        "- Use ONLY the <think> and <answer> tags in your response; do NOT use any other tags (e.g., <tool_call>).\n"
        "- In <think>, perform MULTI-ROUND COMPARISONS across passages as needed.\n"
        "  For each round, compare any pairs and note the winner with a short reason.\n"
        "- Only use lines of the form: 'Round k: Compare [i] vs [j]: [x] is more relevant because ...' (k optional).\n"
        "  Do NOT write 'vs others'/'vs all', do NOT include any full or partial ranking, and do NOT write 'Final order', 'Thus', 'Therefore' in <think>.\n"
        "- In <answer>, output ONLY the FINAL ranking, no extra text.\n"
        "- The final ranking MUST satisfy ALL pairwise outcomes stated in <think> (if [x] beat [y], then [x] must appear before [y]).\n\n"
        "Step 1 — <think> (Comparative reasoning):\n"
        "  a) Make strategic pairwise comparisons to determine relative relevance.\n"
        "  b) For each comparison, use: 'Compare [i] vs [j]: [x] is more relevant because ...'\n"
        "  c) AVOID repeating the same comparison (e.g., if you compared [1] vs [2], don't compare them again).\n"
        "  d) Focus on key comparisons that help establish the ranking efficiently.\n"
        "  e) Ensure there are NO contradictory outcomes (e.g., do not state both [i] > [j] and [j] > [i]); if a conflict arises, resolve it before answering.\n"
        "  f) In <think>, STRICTLY avoid any full or partial ranking text (e.g., lines like 'Final order:', 'Thus:', '[a] > [b] > ...').\n\n"
        "Step 2 — <answer> (Final output only):\n"
        "  - Provide ONLY the final ranking in the form: [X] > [Y] > [Z] > [W] > ...\n"
        "  - Before outputting, VERIFY that the answer respects every 'Compare [i] vs [j]' outcome in <think>.\n"
        "  - Silently verify completeness: include EVERY presented passage identifier exactly once (no omissions, no duplicates).\n"
        "  - NO additional text, explanations, or comments\n"
        "  - NO text outside the <answer> tags\n\n"
        "EXAMPLE FORMAT (illustrative only):\n"
        "<think>\n"
        "Round 1: Compare [2] vs [3]: [3] is more relevant because it addresses the query specifics\n"
        "Round 1: Compare [1] vs [4]: [1] is more relevant because it directly answers the question\n"
        "Round 2: Compare [1] vs [3]: [1] remains more relevant due to coverage and specificity\n"
        "Round 3: Compare [5] vs [6]: [5] is more relevant due to stronger evidence\n"
        "No full ranking provided here.\n"
        "</think>\n\n"
        "<answer>\n"
        "[1] > [3] > [5] > [6] > [2] > [4]\n"
        "</answer>"
    )


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
        "- Your response MUST ALWAYS include an <answer> section before stopping. Do NOT end your response early\n"
        "- After you output </answer>, STOP immediately. Do NOT write anything after </answer>.\n"
        "- In <think>, perform MULTI-ROUND COMPARISONS across passages as needed\n"
        "  For each round, compare any pairs and note the winner with a short reason.\n"
        "- Only use lines of the form: 'Round k: Compare [i] vs [j]: [x] is more relevant because ...' (k optional).\n"
        "- In <answer>, output ONLY the FINAL ranking, no extra text.\n"
        "- The final ranking MUST satisfy ALL pairwise outcomes stated in <think> (if [x] beat [y], then [x] must appear before [y]).\n\n"
        "Step 1 — <think> (Comparative reasoning):\n"
        "  a) Make strategic pairwise comparisons to determine relative relevance.\n"
        "  b) For each comparison, use: 'Compare [i] vs [j]: [x] is more relevant because ...'\n"
        "  c) AVOID repeating the same comparison\n"
        "  d) Focus on key comparisons that help establish the ranking efficiently.\n"
        "  e) Ensure there are NO contradictory outcomes (e.g., do not state both [i] > [j] and [j] > [i]); if a conflict arises, resolve it before answering.\n"
        "Step 2 — <answer> (Final output only):\n"
        "  - Provide ONLY the final ranking in the form: [X] > [Y] > [Z] > [W] > ...\n"
        "  - Before outputting, VERIFY that the answer respects every 'Compare [i] vs [j]' outcome in <think>.\n"
        "  - Silently verify completeness: include EVERY presented passage identifier exactly once (no omissions, no duplicates).\n"
        "  - NO additional text, explanations, or comments\n"
        "  - NO text outside the <answer> tags\n"
        "  - IMPORTANT: Do NOT end your response until you have written the <answer> section. If needed, directly proceed to <answer>.\n"
    )


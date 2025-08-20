#!/usr/bin/env python3
"""
Evaluate Qwen2.5-7B-Instruct on DL19 (qrels_dl19.json + retrieve_results_dl19.json)
using the current multi-round comparison ranking prompt, and report NDCG.

- Loads qrels and retrieval results from JSON files
- For each query, builds a prompt with top-K passages and asks the model to rank them
- Parses <answer> to obtain a full ranking, fills any missing doc indices, reorders hits
- Aggregates results across queries and computes NDCG via trec_eval

Note:
- This script is a test harness; adjust MAX_QUERIES and TOP_K for speed/memory
- Requires: transformers, torch, pytrec_eval
"""
import json
import os
import re
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from rank_prompts import build_elimination_sort_instruction
import pytrec_eval

ROOT = os.path.dirname(os.path.abspath(__file__))
QRELS_PATH = os.path.join(ROOT, 'qrels_dl19.json')
RETRIEVE_PATH = os.path.join(ROOT, 'retrieve_results_dl19.json')

TOKENIZER_NAME = 'Qwen/Qwen2.5-7B-Instruct'
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 10
MAX_QUERIES = 25   # set None to run all
DOC_CHAR_LIMIT = 800  # truncate each doc to avoid huge prompts
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, 'r') as f:
        raw = json.load(f)
    # ensure ints
    return {str(qid): {str(doc): int(rel) for doc, rel in doc_map.items()} for qid, doc_map in raw.items()}


def load_results(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + ' ...'


def build_chat_messages(query: str, passages: List[str]) -> List[Dict[str, str]]:
    # passages are strings with [i] prefix already
    doc_text = "\n".join(passages)
    instruction = build_elimination_sort_instruction(query)

    system_message = (
        "You are an intelligent assistant that ranks passages based on relevance to search queries. "
        "Follow the exact format with <think>/<answer>. "
        "Use ONLY the <think> and <answer> tags; do NOT use any other tags (e.g., <tool_call>). "
        "Your response MUST ALWAYS include an <answer> section; do NOT stop after </think>. "
        "If you are running out of space or are uncertain, SKIP TO <answer> IMMEDIATELY and produce the final ranking. "
        "After you output </answer>, STOP immediately and do NOT write anything else. "
        "Always output a COMPLETE final ranking covering ALL passages exactly once (no omissions, no duplicates)."
    )
    n = len(passages)
    user_message = (
        f"Query: \"{query}\"\n\n"
        f"Passages:\n{doc_text}\n\n"
        f"{instruction}\n\n"
        f"There are {n} passages with identifiers [1]..[{n}]. In <answer>, you MUST include ALL of them exactly once (no omissions, no duplicates).\n"
        f"If you are uncertain about some comparisons, still produce a full order using your best judgment.\n"
        f"Now please produce the final ranking."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
    print('✓ Model loaded')
    return model, tokenizer


def clean_order(raw_indices: List[int], n: int) -> Tuple[List[int], bool, str]:
    """Deduplicate keeping first occurrence, drop out-of-range, and append missing in ascending order.
    Returns (cleaned_order, was_perfect, reason).
    was_perfect=True iff the input already had exactly n unique in-range indices covering all items.
    """
    # preserve order; remove non-integers already handled by caller
    in_range = [i for i in raw_indices if 1 <= i <= n]
    seen = set()
    deduped: List[int] = []
    for i in in_range:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    # append missing in ascending order
    missing = [i for i in range(1, n + 1) if i not in seen]
    cleaned = deduped + missing
    was_perfect = (len(raw_indices) == n and len(deduped) == n and len(missing) == 0)
    reason = "ok" if was_perfect else (
        "duplicates/oor filtered, appended missing in ascending order" if len(missing) > 0 or len(deduped) < len(in_range) or len(in_range) != len(raw_indices) else "normalized"
    )
    return cleaned[:n], was_perfect, reason


def parse_full_order(answer_text: str, n: int) -> Tuple[List[int], Tuple[bool, str]]:
    indices = re.findall(r"\[(\d+)\]", answer_text)
    raw_order = [int(i) for i in indices if i.isdigit()]
    cleaned, was_perfect, reason = clean_order(raw_order, n)
    return cleaned, (was_perfect, reason)
def normalize_response_for_tags(s: str) -> str:
    """Normalize response text to improve <answer> tag detection.
    - Remove zero-width and directional marks
    - Convert fullwidth brackets/angles to ASCII
    - Collapse unusual whitespace around tags
    """
    if not isinstance(s, str):
        return s
    # Remove zero-widths and bidi controls
    zero_widths = [
        "\u200b", "\ufeff", "\u200e", "\u200f",
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    ]
    for zw in zero_widths:
        s = s.replace(zw, "")
    # Fullwidth to ASCII for brackets and angle brackets
    trans = {
        ord("＜"): "<", ord("＞"): ">",
        ord("［"): "[", ord("］"): "]",
        ord("【"): "[", ord("】"): "]",
    }
    s = s.translate(trans)
    # Normalize whitespace around tag names like < answer > -> <answer>
    s = re.sub(r"<\s*(answer|think)\s*>", lambda m: f"<{m.group(1).lower()}>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*(answer|think)\s*>", lambda m: f"</{m.group(1).lower()}>", s, flags=re.IGNORECASE)
    return s



def extract_answer_text_robust(response_text: str) -> str:
    """Extract content from <answer> tag.
    - If both <answer> ... </answer> exist, return inner content.
    - If only opening <answer> exists (no closing), return content from opening tag to end.
    - If no <answer> tag at all, return empty string (trigger fallback to original order).
    """
    if not isinstance(response_text, str):
        return ""
    # Normalize tags and zero-widths to improve detection
    cleaned = normalize_response_for_tags(response_text)
    # Remove tool_call blocks and stray openings
    cleaned = re.sub(r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\s*tool_call\s*>", " ", cleaned, flags=re.IGNORECASE)
    # Prefer content inside <answer> ... </answer>
    m = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If only opening tag is present, take the tail as answer content
    m_open = re.search(r"<\s*answer\s*>", cleaned, re.IGNORECASE)
    if m_open:
        tail = cleaned[m_open.end():]
        return tail.strip()
    return ""


def rank_block(model, tokenizer, query: str, block_hits: List[Dict]) -> List[int]:
    """Ask model to rank a block of passages, must return full order 1..n; otherwise return [].
    Logs the extracted ranking (even if invalid).
    """
    passages = []
    for idx, h in enumerate(block_hits, start=1):
        content = shorten(h.get('content', ''), DOC_CHAR_LIMIT)
        passages.append(f"[{idx}] {content}")

    messages = build_chat_messages(query, passages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Debug: print the full response to see what we got
    print(f"  Full response: {response[:500]}...")

    # Try multiple ways to extract answer content
    answer_content = extract_answer_text_robust(response)
    print("  Raw <answer>:", answer_content)

    n_block = len(block_hits)

    # If we got some ranking content (even without perfect tags), try to parse it
    if answer_content.strip():
        order, (perfect, reason) = parse_full_order(answer_content, n=n_block)
        if order and len(order) == n_block:
            # We got a valid full ranking
            chain = " > ".join([f"[{i}]" for i in order])
            status = "OK" if perfect else f"NORMALIZED ({reason})"
            print(f"  Filtered <answer>: {chain}")
            print(f"  Block ranking: {chain}  => {status}")
            return order

    # Fallback: use original order
    order = list(range(1, n_block + 1))
    chain = " > ".join([f"[{i}]" for i in order])
    print(f"  No valid ranking found. Using original order.")
    print(f"  Filtered <answer>: {chain}")
    print(f"  Block ranking: {chain}  => FALLBACK_TO_ORIGINAL")
    return order


def merge_top10(prev_top10: List[Dict], new_hits: List[Dict], order_indices: List[int]) -> List[Dict]:
    """Given a block (prev_top10 + new10), and the model's order over the 20 items, return new top10.
    prev_top10 and new_hits are lists of hit dicts; order_indices are 1..20.
    """
    block = prev_top10 + new_hits
    n = len(block)
    if not order_indices or len(order_indices) != n or len(set(order_indices)) != n or not all(1 <= i <= n for i in order_indices):
        # invalid, keep previous
        return prev_top10
    # reorder block according to order_indices
    ordered = [block[i-1] for i in order_indices]
    return ordered[:10]


def sliding_window_rank(model, tokenizer, query: str, hits: List[Dict]) -> List[int]:
    """Sliding window over up to 100 passages with window size 20 and stride 10.
    Start with first 10 as top10; then for t in {10,20,...}, rank prev_top10 + next10 and update top10.
    Returns final top10 indices in the original list (1-based indices). Logs each step.
    """
    if not hits:
        return []
    total = min(len(hits), 100)
    # initialize top10 as first 10
    top10 = hits[:10]
    for start in range(10, total, 10):
        end = min(start + 10, total)
        new10 = hits[start:end]
        block = top10 + new10
        # ask model to produce full order of this block
        order = rank_block(model, tokenizer, query, block)
        if not order:
            # keep previous top10
            continue
        top10 = merge_top10(top10, new10, order)
    # return indices mapping back to original hits
    top_docids = set(h['docid'] for h in top10)
    result = []
    for i, h in enumerate(hits[:total], start=1):
        if h['docid'] in top_docids:
            result.append(i)
            if len(result) == 10:
                break
    # If duplicates or fewer than 10 found, pad from start
    seen = set(result)
    for i in range(1, total+1):
        if len(result) >= 10:
            break
        if i not in seen:
            result.append(i)
    return result[:10]


def build_run_results(all_items: List[Dict], all_orders: List[List[int]]) -> Dict[str, Dict[str, float]]:
    """Build run dict for pytrec_eval: {qid: {docid: score}} with descending scores by rank."""
    run: Dict[str, Dict[str, float]] = {}
    for item, order in zip(all_items, all_orders):
        qid = str(item['hits'][0]['qid']) if item['hits'] else None
        if not qid:
            continue
        # Map index -> docid
        top_hits = item['hits'][:TOP_K]
        idx_to_docid = {idx: str(h['docid']) for idx, h in enumerate(top_hits, start=1)}
        # Assign descending scores (higher is better)
        scores = {idx_to_docid[i]: float(TOP_K - rank) for rank, i in enumerate(order, start=1) if i in idx_to_docid}
        if qid not in run:
            run[qid] = {}
        run[qid].update(scores)
    return run


def compute_ndcg(qrels: Dict[str, Dict[str, int]], run: Dict[str, Dict[str, float]], k_values=(1, 5, 10)) -> Dict[str, float]:
    # Build metric names once
    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(run)

    # Aggregate mean across queries
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    num_q = max(1, len(scores))
    for qid, q_scores in scores.items():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += q_scores.get(f"ndcg_cut_{k}", 0.0)
    for k in ndcg:
        ndcg[k] = round(ndcg[k] / num_q, 5)
    return ndcg


def parse_args():
    parser = argparse.ArgumentParser(description='DL19 NDCG with sliding-window rerank by Qwen2.5-7B-Instruct')
    parser.add_argument('--max-queries', type=int, default=43, help='Max queries to evaluate (default: 25); None to run all')
    parser.add_argument('--top-k', type=int, default=10, help='Final top-K to evaluate (default: 10)')
    parser.add_argument('--window', type=int, default=20, help='Sliding window size (default: 20)')
    parser.add_argument('--stride', type=int, default=10, help='Sliding window stride/new docs (default: 10)')
    parser.add_argument('--doc-char-limit', type=int, default=800, help='Max characters per passage (default: 800)')
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='Max new tokens to generate (default: 1024)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature (default: 0.2)')
    return parser.parse_args()


def main():
    args = parse_args()

    global TOP_K, DOC_CHAR_LIMIT, MAX_NEW_TOKENS, TEMPERATURE
    TOP_K = args.top_k
    DOC_CHAR_LIMIT = args.doc_char_limit
    MAX_NEW_TOKENS = args.max_new_tokens
    TEMPERATURE = args.temperature

    # Load data
    qrels_dict = load_qrels(QRELS_PATH)
    retrieve = load_results(RETRIEVE_PATH)

    # Filter only queries with qrels first, then subset to max_queries to ensure we actually run N queries
    total_before = len(retrieve) if retrieve else 0
    overlapped = [it for it in retrieve if it.get('hits') and str(it['hits'][0]['qid']) in qrels_dict] if retrieve else []
    total_overlap = len(overlapped)

    if args.max_queries is not None:
        overlapped = overlapped[:args.max_queries]
    total_eval = len(overlapped)

    print(f"Loaded {total_before} retrieve items; {total_overlap} overlap with qrels; evaluating {total_eval} queries.")

    if not overlapped:
        print('No overlapping queries between retrieve_results and qrels. Exiting.')
        return

    retrieve = overlapped

    # Load model
    model, tokenizer = load_model()

    # Generate rankings per query using sliding window
    all_orders: List[List[int]] = []
    for idx, item in enumerate(retrieve, start=1):
        q = item.get('query', '')
        hits = item.get('hits', [])
        print(f"\n[{idx}/{len(retrieve)}] QID={hits[0]['qid'] if hits else 'N/A'} Query: {q[:120]}...")
        order = sliding_window_rank(model, tokenizer, q, hits)
        print('Model order (indices):', order)
        all_orders.append(order)

    # Build run dict and compute metrics
    run = build_run_results(retrieve, all_orders)
    metrics = compute_ndcg(qrels_dict, run, k_values=(1, 5, 10))

    print("\n" + "=" * 80)
    print('NDCG Results (DL19, Qwen2.5-7B-Instruct, sliding window):')
    for k in (1, 5, 10):
        print(f"  NDCG@{k}: {metrics.get(f'NDCG@{k}', 0.0)}")
    print("=" * 80)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
基于“多轮比较（multi-round comparisons）”提示词的reward函数。
- think 中仅包含多轮比较语句：
  允许前缀 Round X:，核心格式为：
    Compare [i] vs [j]: [x] is more relevant because ...
  其中 x 必须是 i 或 j。
- 检查每一个比较的格式是否正确
- 检查重复比较（同一无序对 {i,j} 多次比较）
- 检查矛盾（同一对 {i,j} 出现不同胜者）或形成逻辑环
- 检查 answer 是否满足 think 中的所有比较结果（若 [x] 胜 [y]，answer 中 x 必须在 y 之前）
- 保留 NDCG 作为主要奖励分量
"""
from __future__ import annotations
import re
import os
import copy
from typing import List, Tuple, Dict, Any, Set

# Expected number of passages required in <answer>
ANSWER_EXPECTED_N = 20

# 直接实现 NDCG 相关工具与计算（不再复用其他文件）

def remove_duplicate(response, permutation):
    """移除重复项"""
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print(f"duplicate {c} in response: {response}")
            print(f"permutation: {permutation}")
    return new_response


def clean_response(response: str):
    """清理响应字符串，保留数字，其余转空格"""
    chars = []
    for c in response:
        if c.isdigit():
            chars.append(c)
        else:
            chars.append(' ')
    return ''.join(chars).strip()


def parse_answer_ids_strict(answer_text: str) -> list:
    """严格从answer文本中提取[数字]格式的ID列表"""
    ids = re.findall(r"\[(\d+)\]", answer_text)
    return [int(x) - 1 for x in ids]  # 转换为0基索引


def receive_permutation(item, answer_text, rank_start=0, rank_end=100):
    """根据answer中的排列对 item['hits'] 进行重排"""
    DEBUG_REWARD = os.getenv("DEBUG_REWARD", "0") == "1"

    if DEBUG_REWARD:
        print(f"original answer: {answer_text}")

    # 严格从answer中提取[数字]格式的排序
    response = parse_answer_ids_strict(answer_text)

    if DEBUG_REWARD:
        print(f"extracted indices: {response}")

    # 如果没有提取到任何有效排序，使用原始顺序
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = list(range(len(cut_range)))

    if not response:
        # 没有answer或answer无效，保持原始排序
        response = original_rank
        if DEBUG_REWARD:
            print("No valid answer found, using original order")
    else:
        # 去重复（保持第一次出现的顺序）
        seen = set()
        deduplicated = []
        for idx in response:
            if idx not in seen and 0 <= idx < len(cut_range):
                seen.add(idx)
                deduplicated.append(idx)

        # 补全缺失的索引（添加到末尾）
        for idx in original_rank:
            if idx not in seen:
                deduplicated.append(idx)

        response = deduplicated

    # 无论是否DEBUG，都打印clean后的顺序（索引与1基序号）
    order_1based = [i + 1 for i in response]
    order_1based_str = " > ".join([f"[{i}]" for i in order_1based])
    print(f"[compare_rounds_reward] clean_order_idx={response}; clean_order={order_1based_str}")

    # 重新排列hits
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def load_qrels_dict(qrels_path: str = None) -> dict:
    """加载 qrels 文件，带多路径回退与环境变量支持"""
    if qrels_path is None:
        env_path = os.environ.get('QRELS_FILE_PATH')
        if env_path and os.path.exists(env_path):
            qrels_path = env_path
        else:
            possible_paths = [
                '/data/coding/Rearank/data/combined_qrels.txt',
                os.path.join(os.path.dirname(__file__), 'data', 'combined_qrels.txt'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_qrels.txt'),
                os.path.join(os.getcwd(), 'data', 'combined_qrels.txt'),
                os.path.join(os.getcwd(), 'combined_qrels.txt'),
                'data/combined_qrels.txt',
                'combined_qrels.txt'
            ]
            for path in possible_paths:
                print(f"Trying qrels path: {path}")
                if os.path.exists(path):
                    qrels_path = path
                    print(f"Found qrels file at: {qrels_path}")
                    break
            if qrels_path is None:
                print("Warning: combined_qrels.txt not found in any of the following paths:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("Using empty qrels_dict")
                return {}

    qrels_dict = {}
    try:
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][docid] = rel
        print(f"Loaded qrels from {qrels_path}")
    except Exception as e:
        print(f"Error loading qrels file {qrels_path}: {e}")
        return {}

    return qrels_dict


def compute_ndcg_reward(predict_str: str, item: dict, qrels_dict: dict = None) -> float:
    """计算 NDCG 奖励，等价于 reward_func.py 中的实现，但不复用导入"""
    try:
        # 尝试多种导入路径
        try:
            from verl.utils.reward_score.trec_eval import eval_rerank
        except ImportError:
            try:
                from Rearank.verl.verl.utils.reward_score.trec_eval import eval_rerank
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'verl'))
                try:
                    from verl.utils.reward_score.trec_eval import eval_rerank
                except Exception:
                    from Rearank.verl.verl.utils.reward_score.trec_eval import eval_rerank

        # 解析出答案里的排列，应用到 item 上
        permutation = predict_str
        item = receive_permutation(item, permutation, rank_start=0, rank_end=len(item['hits']))

        # 确保 qrels_dict 不为 None
        if qrels_dict is None:
            qrels_dict = load_qrels_dict()

        all_metric = eval_rerank('custom', item, qrels_dict=qrels_dict)
        rerank_score = all_metric.get('NDCG@10', 0.0)
        init_score = item['metrics'].get('NDCG@10', 0.0)
        best_score = item['best_metrics'].get('NDCG@10', 0.0)
        score = (rerank_score - init_score) / (best_score - init_score) if best_score != init_score else 0.0
        return score

    except Exception as e:
        print(f"Error in compute_ndcg_reward: {e}")
        return 0.0


# ------------------------ 基础提取 ------------------------

def extract_think_content(predict_str: str) -> str:
    """Extract reasoning text.
    Primary: content inside <think> ... </think>.
    Fallback: if no <think>, use everything before the first <answer> tag; if no <answer> either, use full text.
    This allows scoring when models omit <think> but still write comparison lines in plain text.
    """
    m = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: text before <answer>
    ma = re.search(r"<answer>", predict_str)
    pre = predict_str[: ma.start()] if ma else predict_str
    return pre.strip()


def normalize_response_for_tags(s: str) -> str:
    """Normalize response text to improve <answer> tag detection.
    - Remove zero-width and directional marks
    - Convert fullwidth brackets/angles to ASCII
    - Normalize whitespace around tag names (e.g., < answer > -> <answer>)
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
    # Normalize whitespace around tags like < answer > -> <answer>
    s = re.sub(r"<\s*(answer|think)\s*>", lambda m: f"<{m.group(1).lower()}>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*(answer|think)\s*>", lambda m: f"</{m.group(1).lower()}>", s, flags=re.IGNORECASE)
    return s


def extract_answer_content(predict_str: str) -> str:
    """Extract ONLY the content inside <answer>...
    - If both <answer> and </answer> exist, return the inner content
    - If only opening <answer> exists, return the tail after it
    - If no <answer> tag, return empty string
    """
    cleaned = normalize_response_for_tags(predict_str)
    # Remove tool_call blocks and stray openings to avoid noise
    cleaned = re.sub(r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\s*tool_call\s*>", " ", cleaned, flags=re.IGNORECASE)

    m = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m_open = re.search(r"<\s*answer\s*>", cleaned, re.IGNORECASE)
    if m_open:
        return cleaned[m_open.end():].strip()
    return ""


# ------------------------ 解析比较行 ------------------------

COMPARE_LINE_RE = re.compile(
    r"^\s*(?:Round\s+\d+:\s*)?Compare\s*\[(\d+)\]\s*(?:vs|against|and)\s*\[(\d+)\]\s*:\s*(.*)$",
    re.IGNORECASE,
)
WINNER_RE = re.compile(r"\[(\d+)\]\s*is\s*more\s*relevant\b.*?because\b", re.IGNORECASE)


def parse_comparison_line(line: str) -> Tuple[int | None, int | None, int | None, str | None]:
    """解析一行比较。
    返回 (i, j, winner, reason_text)；若格式不合法，则返回 (None, None, None, None)。
    规则：
      - 必须匹配 COMPARE_LINE_RE
      - 在冒号后的内容中，必须出现 "[x] is more relevant ... because ..."；x 必须等于 i 或 j
    """
    m = COMPARE_LINE_RE.match(line)
    if not m:
        return None, None, None, None
    i, j, tail = int(m.group(1)), int(m.group(2)), m.group(3).strip()

    m2 = WINNER_RE.search(tail)
    if not m2:
        return None, None, None, None
    winner = int(m2.group(1))
    if winner not in (i, j):
        return None, None, None, None

    return i, j, winner, tail


def extract_all_comparisons(think: str) -> Tuple[List[Tuple[int, int, int, str, int]], List[int]]:
    """从 think 中抽取所有比较。
    返回：
      - comparisons: 列表[(i, j, winner, reason_text, line_no)]，i<j 保证无序对一致
      - invalid_lines: 不合法行号列表
    """
    comparisons: List[Tuple[int, int, int, str, int]] = []
    invalid_lines: List[int] = []
    for ln_no, raw in enumerate(think.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        i, j, w, reason = parse_comparison_line(line)
        if i is None:
            # 允许 think 中有其它说明行；仅将以 Compare 开头却不合规的计为 invalid
            if re.match(r"\s*(?:Round\s+\d+:\s*)?Compare\b", line, re.IGNORECASE):
                invalid_lines.append(ln_no)
            continue
        if i == j:
            invalid_lines.append(ln_no)
            continue
        a, b = sorted((i, j))
        winner = w if w in (a, b) else None
        if winner is None:
            invalid_lines.append(ln_no)
            continue
        comparisons.append((a, b, winner, reason, ln_no))
    return comparisons, invalid_lines


# ------------------------ 逻辑检查 ------------------------

def check_duplicates_and_contradictions(
    comparisons: List[Tuple[int, int, int, str, int]]
) -> Tuple[int, int, Dict[Tuple[int, int], int]]:
    """统计重复与矛盾：
    - duplicate_pairs: 对于同一无序对 {a,b}，出现多次比较的次数（>0 表示重复）
    - contradictions: 对于同一对 {a,b}，存在不同胜者的次数（按对统计）
    - winners_by_pair: 记录每对的最终胜者（若矛盾，则以最后一次为准）
    """
    from collections import defaultdict
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    winners: Dict[Tuple[int, int], int] = {}
    contradiction_pairs: Set[Tuple[int, int]] = set()

    for a, b, w, _reason, _ln in comparisons:
        pair = (a, b)
        counts[pair] += 1
        if pair in winners and winners[pair] != w:
            contradiction_pairs.add(pair)
        winners[pair] = w

    duplicate_pairs = sum(1 for k, c in counts.items() if c > 1)
    contradictions = len(contradiction_pairs)
    return duplicate_pairs, contradictions, winners


def has_cycle(edges: List[Tuple[int, int]]) -> bool:
    from collections import defaultdict
    graph: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[int, int] = {}

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in graph.get(u, []):
            c = color.get(v, WHITE)
            if c == GRAY:
                return True
            if c == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for node in list(graph.keys()):
        if color.get(node, WHITE) == WHITE:
            if dfs(node):
                return True
    return False


# ------------------------ Answer 检查 ------------------------

def parse_answer_order(answer: str) -> List[int]:
    ids = re.findall(r"\[(\d+)\]", answer)
    return [int(x) for x in ids]


def check_answer_respects_comparisons(answer: str, winners_by_pair: Dict[Tuple[int, int], int]) -> Tuple[float, int, int]:
    """检查 answer 是否满足比较结论。
    返回 (比例分数, 满足条数, 总约束数)。
    若无比较，返回 (1.0, 0, 0)。
    """
    order = parse_answer_order(answer)
    if not winners_by_pair:
        return 1.0, 0, 0
    if not order:
        return 0.0, 0, len(winners_by_pair)

    pos = {doc: idx for idx, doc in enumerate(order)}
    total = 0
    satisfied = 0
    for (a, b), w in winners_by_pair.items():
        # 决定 loser
        loser = b if w == a else a
        total += 1
        if w in pos and loser in pos and pos[w] < pos[loser]:
            satisfied += 1
    return (satisfied / total if total else 1.0), satisfied, total


# ------------------------ 汇总评分 ------------------------

def compute_compare_rounds_reward(predict_str: str, item: dict, qrels_dict: dict | None = None) -> dict:
    think = extract_think_content(predict_str)
    answer = extract_answer_content(predict_str)

    # 1) 解析比较（仅用于统计格式和重复）
    comparisons, invalid_lines = extract_all_comparisons(think)

    # 2) 统计重复（不再检查 answer 一致性与是否成环）
    dup_pairs, _contradiction_pairs, _winners_by_pair = check_duplicates_and_contradictions(comparisons)

    # 3) 严格答案格式：必须是 [1..ANSWER_EXPECTED_N] 的一个排列（无重复、无遗漏、无越界）
    def parse_answer_ids(ans: str) -> List[int]:
        ids = re.findall(r"\[(\d+)\]", ans)
        return [int(x) for x in ids]

    answer_ids = parse_answer_ids(answer)
    expected_set = set(range(1, ANSWER_EXPECTED_N + 1))
    answer_set = set(answer_ids)
    correct_len = len(answer_ids) == ANSWER_EXPECTED_N
    no_duplicates = len(answer_ids) == len(answer_set)
    in_range = answer_set.issubset(expected_set)
    answer_format = 1.0 if (correct_len and no_duplicates and in_range) else 0.0

    # 4) NDCG：始终基于answer解析；若answer无效则采用原始排序（在receive_permutation内部处理）
    ndcg_reward = 0.0
    if compute_ndcg_reward is not None:
        ndcg_reward = compute_ndcg_reward(answer, item, qrels_dict)

    # 5) think 格式正确性：所有以 Compare 开头的行都能被解析
    compare_lines = [ln for ln in think.splitlines() if re.match(r"\s*(?:Round\s+\d+:\s*)?Compare\b", ln.strip(), re.IGNORECASE)]
    total_compare_lines = len(compare_lines)
    valid_compare_lines = len(comparisons)
    think_format = (valid_compare_lines / total_compare_lines) if total_compare_lines else (1.0 if not think else 0.0)

    # 6) think 无重复比较
    think_no_duplicates = 1.0 if dup_pairs == 0 else 0.0

    # 7) 组合评分：ndcg 0.7 + think_format 0.1 + answer_format 0.1 + think_no_duplicates 0.1
    total_score = (
        0.7 * ndcg_reward +
        0.1 * think_format +
        0.1 * answer_format +
        0.1 * think_no_duplicates
    )
    print(
        f"[compare_rounds_reward] ndcg={ndcg_reward:.6f}, "
        f"think_format={think_format:.6f}, "
        f"answer_format={answer_format:.6f}, "
        f"think_no_duplicates={think_no_duplicates:.6f}, "
        f"total={total_score:.6f}"
    )

    return {
        "score": total_score,
        "ndcg_reward": ndcg_reward,
        "think_format": think_format,
        "answer_format": answer_format,
        "think_no_duplicates": think_no_duplicates,
        "duplicate_pairs": dup_pairs,
        "invalid_compare_lines": invalid_lines,
        "num_valid_comparisons": valid_compare_lines,
        "num_compare_lines": total_compare_lines,
        "has_think_answer": 1.0 if (think and answer) else 0.0,
    }


# 主要入口，供 verl 调用

def compute_score(predict_str: str, item: dict, qrels_dict: dict) -> dict:
    return compute_compare_rounds_reward(predict_str, item, qrels_dict)


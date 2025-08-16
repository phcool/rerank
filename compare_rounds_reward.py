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
from typing import List, Tuple, Dict, Any, Set

# 复用NDCG计算（避免重复代码）
try:
    from elimination_sort_reward import compute_ndcg_reward
except Exception:
    compute_ndcg_reward = None


# ------------------------ 基础提取 ------------------------

def extract_think_content(predict_str: str) -> str:
    m = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_answer_content(predict_str: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    return m.group(1).strip() if m else ""


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

    # 1) 解析比较
    comparisons, invalid_lines = extract_all_comparisons(think)

    # 2) 重复与矛盾
    dup_pairs, contradiction_pairs, winners_by_pair = check_duplicates_and_contradictions(comparisons)

    # 3) 环检测（由 winners_by_pair 导出的有向边）
    edges = []
    for (a, b), w in winners_by_pair.items():
        loser = b if w == a else a
        edges.append((w, loser))
    cycle = has_cycle(edges) if edges else False

    # 4) 答案一致性
    ans_consistency, satisfied_cnt, total_constraints = check_answer_respects_comparisons(answer, winners_by_pair)

    # 5) 答案格式
    answer_format = 1.0 if re.match(r"^\s*\[\d+\](\s*>\s*\[\d+\])*\s*$", answer.strip()) else 0.0 if answer else 0.0

    # 6) NDCG
    ndcg_reward = 0.0
    if compute_ndcg_reward is not None:
        ndcg_reward = compute_ndcg_reward(predict_str, item, qrels_dict)

    # 7) 比较格式覆盖率（每个 Compare 行均匹配）
    compare_lines = [ln for ln in think.splitlines() if re.match(r"\s*(?:Round\s+\d+:\s*)?Compare\b", ln.strip(), re.IGNORECASE)]
    total_compare_lines = len(compare_lines)
    valid_compare_lines = len(comparisons)
    format_cover = (valid_compare_lines / total_compare_lines) if total_compare_lines else (1.0 if not think else 0.0)

    # 8) 组合评分（总和约1.0）
    total_score = (
        0.5 * ndcg_reward +
        0.2 * ans_consistency +
        0.1 * format_cover +
        0.1 * answer_format +
        0.1 * (0.0 if cycle else 1.0)
    )

    # 对重复与矛盾做扣分
    penalty = 0.0
    if dup_pairs > 0:
        penalty += min(0.1, 0.05 * dup_pairs)
    if contradiction_pairs > 0:
        penalty += min(0.2, 0.1 * contradiction_pairs)
    total_score = max(0.0, total_score - penalty)

    return {
        "score": total_score,
        "ndcg_reward": ndcg_reward,
        "answer_consistency": ans_consistency,
        "answer_format": answer_format,
        "format_cover": format_cover,
        "duplicate_pairs": dup_pairs,
        "contradiction_pairs": contradiction_pairs,
        "has_cycle": cycle,
        "satisfied_constraints": satisfied_cnt,
        "total_constraints": total_constraints,
        "invalid_compare_lines": invalid_lines,
        "num_valid_comparisons": valid_compare_lines,
        "num_compare_lines": total_compare_lines,
        "has_think_answer": 1.0 if (think and answer) else 0.0,
    }


# 主要入口，供 verl 调用

def compute_score(predict_str: str, item: dict, qrels_dict: dict) -> dict:
    return compute_compare_rounds_reward(predict_str, item, qrels_dict)


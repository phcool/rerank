#!/usr/bin/env python3
"""
淘汰排序格式检查的自定义reward函数
"""
import re
import copy
import os
from typing import Dict, Any


def extract_think_content(predict_str: str) -> str:
    """提取<think>标签中的内容"""
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    return think_match.group(1).strip() if think_match else ""


def extract_answer_content(predict_str: str) -> str:
    """提取<answer>标签中的内容"""
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    return answer_match.group(1).strip() if answer_match else ""


def check_elimination_sort_format(think_content: str) -> float:
    """
    检查think部分是否符合淘汰排序格式要求
    
    检查项目：
    1. 是否有起始段落声明
    2. 是否有Round标记
    3. 是否有淘汰理由格式 "Round X: [id] is least relevant because [reason]"
    4. 是否有剩余段落状态
    5. 是否有完成声明和最终排序
    """
    if not think_content:
        return 0.0
    
    score = 0.0
    max_score = 5.0
    
    # 1. 检查起始段落声明 (1分)
    starting_patterns = [
        r"Starting passages:\s*\[.*?\]",
        r"Start with.*:\s*\[.*?\]",
        r"Initial.*:\s*\[.*?\]"
    ]
    for pattern in starting_patterns:
        if re.search(pattern, think_content, re.IGNORECASE):
            score += 1.0
            break
    
    # 2. 检查Round标记 (1分)
    if re.search(r"Round\s+\d+:", think_content, re.IGNORECASE):
        score += 1.0
    
    # 3. 检查淘汰理由格式 (1分)
    elimination_pattern = r"Round\s+\d+:\s*\[\d+\]\s*is\s+least\s+relevant\s+because"
    if re.search(elimination_pattern, think_content, re.IGNORECASE):
        score += 1.0
    
    # 4. 检查剩余段落状态 (1分)
    if re.search(r"Remaining:\s*\[.*?\]", think_content, re.IGNORECASE):
        score += 1.0
    
    # 5. 检查完成声明 (1分)
    completion_patterns = [
        r"Elimination complete",
        r"Final ranking.*reverse.*elimination",
        r"reverse.*order.*elimination"
    ]
    for pattern in completion_patterns:
        if re.search(pattern, think_content, re.IGNORECASE):
            score += 1.0
            break
    
    return score / max_score


def check_elimination_consistency(think_content: str, answer_content: str) -> float:
    """
    检查answer部分是否与think部分的淘汰过程一致
    如果不一致则返回0分
    """
    if not think_content or not answer_content:
        return 0.0

    # 提取淘汰顺序
    elimination_rounds = re.findall(r"Round\s+\d+:\s*\[(\d+)\]", think_content, re.IGNORECASE)

    if not elimination_rounds:
        return 0.0

    # 从answer中提取排序结果
    answer_numbers = re.findall(r'\[(\d+)\]', answer_content)

    if not answer_numbers:
        return 0.0

    # 找出所有提到的文档ID
    all_docs = set(elimination_rounds + answer_numbers)

    # 构建期望的完整排序
    # 淘汰顺序的逆序 + 未被淘汰的文档（最相关的）
    eliminated_docs = elimination_rounds
    remaining_docs = [doc for doc in all_docs if doc not in eliminated_docs]

    # 最终排序：剩余文档（最相关）+ 淘汰顺序的逆序
    expected_ranking = remaining_docs + list(reversed(eliminated_docs))

    # 检查是否完全一致，不一致则返回0分
    if expected_ranking == answer_numbers:
        return 1.0
    else:
        return 0.0


def check_elimination_logic(think_content: str) -> float:
    """
    检查淘汰逻辑的合理性
    """
    if not think_content:
        return 0.0
    
    score = 0.0
    max_score = 3.0
    
    # 1. 检查是否每轮都有理由说明 (1分)
    rounds = re.findall(r"Round\s+\d+:", think_content, re.IGNORECASE)
    reasons = re.findall(r"because\s+[^\\n]+", think_content, re.IGNORECASE)
    
    if len(rounds) > 0 and len(reasons) >= len(rounds) * 0.8:  # 至少80%的轮次有理由
        score += 1.0
    
    # 2. 检查剩余段落数量是否递减 (1分)
    remaining_matches = re.findall(r"Remaining:\s*\[(.*?)\]", think_content, re.IGNORECASE)
    if len(remaining_matches) >= 2:
        # 检查剩余数量是否递减
        remaining_counts = []
        for match in remaining_matches:
            count = len(re.findall(r'\d+', match))
            remaining_counts.append(count)
        
        # 检查是否严格递减
        is_decreasing = all(remaining_counts[i] > remaining_counts[i+1] 
                          for i in range(len(remaining_counts)-1))
        if is_decreasing:
            score += 1.0
    
    # 3. 检查是否有重复淘汰 (1分)
    eliminated_ids = re.findall(r"Round\s+\d+:\s*\[(\d+)\]", think_content, re.IGNORECASE)
    if len(eliminated_ids) == len(set(eliminated_ids)):  # 无重复
        score += 1.0
    
    return score / max_score


def check_answer_format(answer_content: str) -> float:
    """检查answer部分的格式是否正确"""
    if not answer_content:
        return 0.0
    
    # 检查是否符合 [X] > [Y] > [Z] 格式
    pattern = r'^\s*\[\d+\](\s*>\s*\[\d+\])*\s*$'
    if re.match(pattern, answer_content.strip()):
        return 1.0
    return 0.0


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
    """清理响应字符串"""
    chars = []
    for c in response:
        if c.isdigit():
            chars.append(c)
        else:
            chars.append(' ')
    return ''.join(chars).strip()


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    """处理排列"""
    print(f"original answer: {permutation}")
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response, permutation)
    print(f"cleaned answer: {response}")
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    print(f"final permutation index: {response}")
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def load_qrels_dict(qrels_path: str = None) -> dict:
    """加载qrels文件"""
    if qrels_path is None:
        # 首先检查环境变量
        env_path = os.environ.get('QRELS_FILE_PATH')
        if env_path and os.path.exists(env_path):
            qrels_path = env_path
        else:
            # 尝试多个可能的路径
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
    """计算NDCG reward"""
    try:
        # 尝试多种导入路径
        try:
            from verl.utils.reward_score.trec_eval import eval_rerank
        except ImportError:
            try:
                from Rearank.verl.verl.utils.reward_score.trec_eval import eval_rerank
            except ImportError:
                # 如果都失败，使用相对导入
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), 'verl', 'verl', 'utils', 'reward_score'))
                from trec_eval import eval_rerank

        print('-'*50, f"query: {item['query']}", '-'*50)
        print(f"full response: {predict_str}")
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        permutation = content_match.group(1).strip() if content_match else predict_str.strip()

        item = receive_permutation(item, permutation, rank_start=0, rank_end=len(item['hits']))
        # 确保qrels_dict不为None，避免trec_eval尝试加载文件
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


def compute_elimination_sort_reward(predict_str: str, item: dict, qrels_dict: dict = None) -> dict:
    """
    计算淘汰排序相关的reward

    Args:
        predict_str: LLM的预测输出
        item: 数据项
        qrels_dict: 相关性判断字典

    Returns:
        包含各种reward分数的字典
    """
    # 如果qrels_dict为空，尝试加载
    if qrels_dict is None:
        qrels_dict = load_qrels_dict()

    # 提取think和answer内容
    think_content = extract_think_content(predict_str)
    answer_content = extract_answer_content(predict_str)

    # 计算各项分数
    elimination_format_score = check_elimination_sort_format(think_content)
    elimination_consistency_score = check_elimination_consistency(think_content, answer_content)
    elimination_logic_score = check_elimination_logic(think_content)
    answer_format_score = check_answer_format(answer_content)

    # 基础格式检查
    has_think_answer = 1.0 if (think_content and answer_content) else 0.0

    # 计算NDCG reward (使用原有逻辑)
    ndcg_reward = compute_ndcg_reward(predict_str, item, qrels_dict)

    # 综合分数计算
    # 权重分配：NDCG 50%, 淘汰格式 20%, 淘汰逻辑 15%, 一致性 10%, 答案格式 5%
    total_score = (
        0.5 * ndcg_reward +
        0.2 * elimination_format_score +
        0.15 * elimination_logic_score +
        0.1 * elimination_consistency_score +
        0.05 * answer_format_score
    )

    return {
        "score": total_score,
        "ndcg_reward": ndcg_reward,
        "elimination_format_score": elimination_format_score,
        "elimination_consistency_score": elimination_consistency_score,
        "elimination_logic_score": elimination_logic_score,
        "answer_format_score": answer_format_score,
        "has_think_answer": has_think_answer
    }


# 主要的reward函数，供verl调用
def compute_score(predict_str: str, item: dict, qrels_dict: dict = None) -> dict:
    """
    主要的reward计算函数，供verl框架调用
    """
    return compute_elimination_sort_reward(predict_str, item, qrels_dict)

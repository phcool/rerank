#!/usr/bin/env python3
import os
from typing import Optional, List, Dict, Any

from datasets import load_dataset, Dataset


def save_dataset(dataset: Dataset, output_dir: str, filename: str, save_format: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if save_format == "parquet":
        if not output_path.endswith(".parquet"):
            output_path += ".parquet"
        dataset.to_parquet(output_path)
    elif save_format == "jsonl":
        if not output_path.endswith(".jsonl"):
            output_path += ".jsonl"
        dataset.to_json(output_path, lines=True, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported save_format: {save_format}. Use 'parquet' or 'jsonl'.")

    return output_path


def build_elimination_sort_instruction(query_text: Optional[str]) -> str:
    q = f'"{query_text}"' if query_text else "the search query"
    return (
        f"Please rank these passages according to their relevance to the search query: {q}\n\n"
        "CRITICAL: You MUST follow these steps EXACTLY. Do NOT deviate from this format.\n\n"
        "Step 1: Within <think> tags, perform ELIMINATION SORTING:\n"
        "   a) Start with all passages: [1], [2], [3], ..., [n]\n"
        "   b) For each round, identify the LEAST relevant passage among remaining ones:\n"
        "      - Compare ALL remaining passages to the query\n"
        "      - Identify which passage is LEAST relevant and explain why\n"
        "      - State: 'Round X: [passage_id] is least relevant because [reason]'\n"
        "      - Remove it from consideration\n"
        "   c) Show remaining passages after each elimination: 'Remaining: [a], [b], [c], ...'\n"
        "   d) Continue until only one passage remains\n"
        "   e) The final ranking is the REVERSE order of elimination (last eliminated = most relevant)\n\n"
        "Step 2: Within <answer> tags, provide ONLY the final ranking:\n"
        "   - Format: [X] > [Y] > [Z] > [W] > ...\n"
        "   - Most relevant first, least relevant last\n"
        "   - Include ALL passage identifiers EXACTLY ONCE\n"
        "   - NO additional text, explanations, or comments\n"
        "   - NO text outside the <answer> tags\n\n"
        "EXAMPLE FORMAT:\n"
        "<think>\n"
        "Starting passages: [1], [2], [3], [4]\n"
        "Round 1: [4] is least relevant because it doesn't address the query topic\n"
        "Remaining: [1], [2], [3]\n"
        "Round 2: [2] is least relevant because it's less specific than others\n"
        "Remaining: [1], [3]\n"
        "Round 3: [3] is least relevant because [1] provides more comprehensive information\n"
        "Remaining: [1]\n"
        "Elimination complete. Final ranking (reverse of elimination): [1] > [3] > [2] > [4]\n"
        "</think>\n\n"
        "<answer>\n"
        "[1] > [3] > [2] > [4]\n"
        "</answer>"
    )


def transform_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    prompts = example.get("prompt")
    if not isinstance(prompts, list):
        return {"prompt": prompts}

    # 构造新的尾部指令
    new_tail = build_elimination_sort_instruction(example.get("query"))

    # 找到最后一个用户侧的排序指令消息并替换，否则追加一个新的
    target_idx = None
    for i in range(len(prompts) - 1, -1, -1):
        msg = prompts[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = str(msg.get("content", ""))
            if (
                "Please rank these passages" in content
                or "Follow these steps" in content
                or "Rank the passages" in content
            ):
                target_idx = i
                break

    new_prompts: List[Dict[str, str]] = list(prompts)
    if target_idx is None:
        new_prompts.append({"role": "user", "content": new_tail})
    else:
        new_prompts[target_idx] = {"role": "user", "content": new_tail}

    return {"prompt": new_prompts}



def main() -> None:
    # 直接设置参数值
    dataset_id = "le723z/rearank_12k"
    config = None
    split = "train"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rearank_12k")
    save_format = "parquet"
    hf_token = None

    try:
        # 尝试不同的加载方式来解决Feature type List不支持的问题
        print(f"正在下载数据集: {dataset_id}")

        # 方法1: 尝试指定trust_remote_code=True
        try:
            ds = load_dataset(dataset_id, name=config, split=split, token=hf_token)
        except Exception as e1:
            print(f"方法1失败: {e1}")

            # 方法2: 尝试设置download_mode="force_redownload"
            try:
                ds = load_dataset(dataset_id, name=config, split=split, token=hf_token, download_mode="force_redownload")
            except Exception as e2:
                print(f"方法2失败: {e2}")

                # 方法3: 尝试streaming模式然后转换
                try:
                    print("尝试使用流式模式...")
                    ds_streaming = load_dataset(dataset_id, name=config, split=split, token=hf_token, streaming=True)

                    # 将流式数据集转换为普通数据集
                    data_list = []
                    for i, example in enumerate(ds_streaming):
                        data_list.append(example)
                        if i % 1000 == 0:
                            print(f"已处理 {i} 条数据...")
                        # 限制数据量避免内存问题
                        if i >= 10000:  # 只取前10000条数据
                            print("数据量较大，只取前10000条")
                            break

                    ds = Dataset.from_list(data_list)
                    print(f"成功转换 {len(ds)} 条数据")

                except Exception as e3:
                    print(f"方法3失败: {e3}")
                    raise e3

    except Exception as e:
        raise SystemExit(
            f"下载失败：{e}\n"
            f"请确认数据集是否存在且可访问：'{dataset_id}'（config: {config}, split: {split}）。\n"
            f"如该 ID 不存在，请通过代码修改指定正确的数据集 ID。\n"
            f"也可能是datasets库版本问题，请尝试升级: pip install -U datasets pyarrow"
        )

    # 推断默认输出文件名
    repo_name = dataset_id.split("/")[-1]
    config_name = config if config else "default"
    output_name = f"{repo_name}__{config_name}__{split}"

    # 1) 保存原始数据（未修改 prompt）
    original_path = save_dataset(ds, output_dir, output_name, save_format)
    print(f"已保存原始数据到: {original_path}")

    # 2) 重写 prompt 列为淘汰排序算法的格式，并另存
    print("正在重写 prompt 列为淘汰排序算法的格式...")
    ds_elimination_sort = ds.map(transform_prompt, desc="rewrite_prompt_elimination_sorting")
    output_name_elimination_sort = f"{output_name}__elimination_sort"
    elimination_sort_path = save_dataset(ds_elimination_sort, output_dir, output_name_elimination_sort, save_format)
    print(f"已保存淘汰排序版到: {elimination_sort_path}")


if __name__ == "__main__":
    main()
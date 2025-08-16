#!/usr/bin/env python3
"""
使用与produce_data.py一致的“灵活排序”prompt，构造1个query和10个doc，
用 Qwen2.5-7B-Instruct 进行回答，并输出<think>/<answer>部分。

注：为避免对datasets等依赖的导入需求，本测试脚本内复刻了
produce_data.py中的 build_elimination_sort_instruction 文本逻辑，确保prompt一致。
"""
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_prompts import build_elimination_sort_instruction
import compare_rounds_reward as cr_reward


def create_test_data():
    query = "what are the health benefits of regular exercise"
    documents = [
        "[1] Regular exercise helps maintain a healthy weight and reduces the risk of obesity-related diseases. Sustained caloric expenditure and improved insulin sensitivity are frequently cited mechanisms. Large cohort studies associate higher activity with lower BMI and metabolic syndrome prevalence. Randomized lifestyle trials show clinically meaningful reductions in HbA1c among participants who increase weekly activity. Mendelian randomization analyses further suggest a causal link between higher physical activity and lower adiposity.",
        "[2] Physical activity can improve mental health by reducing symptoms of depression and anxiety. Exercise triggers endorphin release and may regulate stress pathways such as HPA axis activity. Randomized controlled trials show moderate effects on mood and perceived stress. Meta-analyses report effect sizes comparable to first-line psychotherapy in mild-to-moderate cases, with low risk of adverse events. Group-based programs may add benefits via social connectedness and routine building.",
        "[3] Exercise strengthens the cardiovascular system, lowering blood pressure and improving heart health. Aerobic training enhances stroke volume and endothelial function while reducing LDL cholesterol. Cardiorespiratory fitness is a strong predictor of all-cause mortality. Large meta-analyses show reductions in major adverse cardiovascular events with structured aerobic and resistance training. Hypertensive and prehypertensive adults often experience clinically significant systolic and diastolic reductions after 8–12 weeks.",
        "[4] Many people enjoy watching sports on television during their free time. Although entertaining, passive viewing does not confer the physiological adaptations associated with exercise. It can even displace time that would otherwise be used for physical activity. Higher sedentary time is consistently associated with increased cardiometabolic risk, independent of moderate-to-vigorous activity. Snack consumption during screen time can also worsen energy balance.",
        "[5] Regular physical activity enhances immune system function and helps prevent common illnesses. Moderate exercise is linked to improved surveillance by natural killer cells and reduced systemic inflammation. Observational evidence suggests fewer upper respiratory infections among active adults. Exercise may augment vaccine responses in older adults by mitigating immunosenescence. The relationship appears J-shaped, with extremely strenuous unaccustomed bouts transiently elevating infection risk.",
        "[6] Exercise improves bone density and muscle strength, reducing the risk of osteoporosis and falls. Weight-bearing and resistance training stimulate osteogenesis and neuromuscular coordination. Benefits are pronounced in older adults when combined with adequate protein and vitamin D. Progressive overload and balance training together can reduce fracture risk by lowering fall incidence. Trials report femoral neck BMD gains after months of targeted resistance exercise.",
        "[7] Regular stretching and exercise can increase flexibility and range of motion. Mobility routines may reduce stiffness and improve functional movement patterns. While less central to disease risk, flexibility can support injury prevention and comfortable daily living. Programs that combine dynamic stretching with light strengthening often yield better functional outcomes than stretching alone. Some yoga and Pilates trials show small-to-moderate benefits for pain and function.",
        "[8] Proper hydration strategies are essential for marathon runners to avoid cramps and fatigue. This guidance is specific to endurance events rather than general health outcomes. It does not replace the broad preventive effects of routine physical activity on chronic disease. Overhydration and underhydration both carry risks (e.g., hyponatremia or heat illness) in long races. These niche considerations are not central to population-level exercise benefits.",
        "[9] Consistent exercise can improve sleep quality and daytime energy levels. Physical activity influences circadian regulation and sleep architecture, often increasing slow-wave sleep. Better sleep in turn supports metabolic health and cognitive performance. Clinical studies in insomnia show small-to-moderate improvements with aerobic or combined training. Morning or early-afternoon sessions may optimize phase-shifting for certain chronotypes.",
        "[10] Some people collect stamps as a hobby unrelated to physical activity. This pastime is not associated with improvements in metabolic, cardiovascular, or musculoskeletal health and therefore is irrelevant to exercise benefits. Although cognitively engaging, it does not induce the physiological adaptations seen with regular training. Any health value would be indirect (e.g., stress relief) and not comparable to exercise’s established effects.",
    ]
    return query, documents


def build_chat_messages(query: str, documents: list[str]):
    # 将文档合并
    doc_text = "\n".join(documents)

    # 复用（与produce_data一致的）指令
    instruction = build_elimination_sort_instruction(query)

    # 采用对话格式，方便Qwen-Instruct使用
    system_message = (
        "You are an intelligent assistant that ranks passages based on relevance to search queries. "
        "Follow the exact format with <think>/<answer>. "
        "Use ONLY the <think> and <answer> tags; do NOT use any other tags (e.g., <tool_call>)."
    )

    # 用户消息先给出query和passages，再给出指令（确保模型知道上下文后再按指令推理）
    user_message = (
        f"Query: \"{query}\"\n\n"
        f"Passages:\n{doc_text}\n\n"
        f"{instruction}\n\n"
        f"Before writing <answer>, check completeness: if there are 10 passages [1]..[10], the answer must contain all [1]..[10] exactly once.\n"
        f"Now please produce the final ranking."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def load_qwen_model():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
    print("✓ Model loaded")
    return model, tokenizer


def generate_response(model, tokenizer, messages: list[dict]):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            early_stopping=True,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # 截断到</answer>
    if "</answer>" in response:
        end_pos = response.find("</answer>") + len("</answer>")
        response = response[:end_pos]
    return response.strip()


def analyze_response(response: str, query: str, documents: list[str]):
    print("\n" + "="*80)
    print("灵活排序 模型响应分析")
    print("="*80)
    print(f"完整响应:\n{response}")
    print("\n" + "-"*60)
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    print(f"Think部分:\n{think}")
    print(f"\nAnswer部分:\n{answer}")
    if answer:
        ranking = re.findall(r"\[(\d+)\]", answer)
        if ranking:
            print("\n最终排序结果:")
            for i, doc_id in enumerate(ranking, 1):
                idx = int(doc_id) - 1
                if 0 <= idx < len(documents):
                    print(f"  {i}. {documents[idx]}")


def build_fake_item_for_reward(query: str, documents: list[str], qid: str = "Q1") -> dict:
    # 构造一个与NDCG计算兼容的最小item结构（必须包含 qid 和 docid）
    hits = []
    for i in range(1, len(documents) + 1):
        hits.append({
            'qid': qid,
            'docid': f'D{i}',
            'rank': i,
            'score': float(len(documents) - i)  # 任意分数，占位
        })
    item = {
        'query': query,
        'hits': hits,
        'metrics': {'NDCG@10': 0.0},
        'best_metrics': {'NDCG@10': 1.0},
    }
    return item


def print_reward_summary(reward: dict):
    print("\n" + "-"*80)
    print("Reward 评估结果")
    print("-"*80)
    # 关键指标
    keys_of_interest = [
        'score', 'ndcg_reward', 'answer_consistency', 'answer_format', 'format_cover',
        'duplicate_pairs', 'contradiction_pairs', 'has_cycle',
        'satisfied_constraints', 'total_constraints', 'invalid_compare_lines',
        'num_valid_comparisons', 'num_compare_lines'
    ]
    for k in keys_of_interest:
        if k in reward:
            print(f"{k}: {reward[k]}")
    # 其余信息简要打印
    extra = {k: v for k, v in reward.items() if k not in keys_of_interest}
    if extra:
        print("\n(更多明细)")
        print(json.dumps(extra, ensure_ascii=False, indent=2))


def build_gold_qrels() -> tuple[dict, list[int]]:
    """构造单-query 的理想相关性，用于NDCG。
    返回 qrels_dict 以及理想顺序（按文档编号）。
    约定 docid = D{i} 与文档 [i] 对应。
    评分策略：
      - Top3: 3 分；Next3: 2 分；Next2: 1 分；Last2: 0 分。
    """
    gold_order = [1, 3, 5, 6, 2, 9, 7, 8, 4, 10]
    grades = [3, 3, 3, 2, 2, 2, 1, 1, 0, 0]
    qrels = {"Q1": {}}
    for i, rel in zip(gold_order, grades):
        qrels["Q1"][f"D{i}"] = rel
    return qrels, gold_order


def main():
    print("=== Qwen2.5-7B 灵活排序 Prompt 测试（10文档+NDCG） ===\n")
    # 1) 准备数据
    query, documents = create_test_data()
    print(f"Query: {query}")
    print("Documents:")
    for d in documents:
        print("  " + d)

    # 打印理想顺序
    qrels_dict, gold_order = build_gold_qrels()
    print("\nGold ranking (by doc index):", " > ".join([f"[{i}]" for i in gold_order]))

    # 2) 构建消息（使用与produce_data相同的指令文本）
    messages = build_chat_messages(query, documents)

    # 3) 加载模型并生成
    try:
        model, tokenizer = load_qwen_model()
        response = generate_response(model, tokenizer, messages)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        print("请确保已安装 transformers、torch，并有足够内存/GPU。")
        return

    # 4) 简单分析输出
    analyze_response(response, query, documents)

    # 5) 构造item并计算reward
    item = build_fake_item_for_reward(query, documents, qid="Q1")
    try:
        reward = cr_reward.compute_score(response, item, qrels_dict=qrels_dict)
    except Exception as e:
        print(f"❌ 计算reward失败: {e}")
        return
    print_reward_summary(reward)

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()


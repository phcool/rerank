#!/usr/bin/env python3
"""
使用与produce_data.py一致的“灵活排序”prompt，构造1个query和20个doc，
用 Qwen2.5-7B-Instruct 进行回答，并输出<think>/<answer>部分。

注：为避免对datasets等依赖的导入需求，本测试脚本内复刻了
produce_data.py中的 build_elimination_sort_instruction 文本逻辑，确保prompt一致。
"""
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_prompts import build_elimination_sort_instruction


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
        "[11] Moderate-intensity aerobic exercise demonstrates HbA1c reductions in adults with type 2 diabetes across multiple randomized trials. Combined diet-and-exercise interventions often produce larger effects on glycemic control, and benefits persist with long-term adherence. Mechanisms include increased glucose transport and improved mitochondrial efficiency in skeletal muscle.",
        "[12] Regular physical activity supports cognitive health, with evidence for improved executive function and processing speed. Aerobic and resistance training may increase hippocampal volume and neurotrophic factors (e.g., BDNF). Prospective cohorts suggest reduced risk of dementia among physically active individuals, controlling for education and baseline health.",
        "[13] Workplace and community-based exercise programs can reduce musculoskeletal pain, especially low back and neck pain. Strengthening and mobility protocols tailored to occupational demands show modest but meaningful improvements in function and presenteeism. Adherence and proper progression are key to sustained benefit.",
        "[14] High-intensity interval training (HIIT) efficiently improves VO2max and insulin sensitivity. Shorter sessions with appropriate recovery can deliver cardiometabolic benefits comparable to longer moderate-intensity workouts, when supervised and progressed safely. HIIT may not suit all populations without medical screening.",
        "[15] Progressive resistance training increases lean mass and improves resting metabolic rate, supporting better weight management. It also enhances glucose uptake via GLUT4 expression and improves bone loading. Programs should individualize load, volume, and frequency for safety and adherence.",
        "[16] Breaking up prolonged sitting with brief activity bouts can improve postprandial glucose and lipid responses, independent of formal exercise sessions. These “sedentary breaks” offer an accessible strategy alongside planned workouts to improve cardiometabolic risk profiles.",
        "[17] Outdoor physical activity can combine light-to-moderate exercise with sun exposure, potentially improving vitamin D status while supporting mood. Although indirect, these factors may complement structured exercise for overall well-being.",
        "[18] Team-based sports provide social connection and motivation that can enhance adherence to exercise routines. While the social benefits are indirect, improved consistency contributes to realized health outcomes over time.",
        "[19] Balance and proprioceptive training reduce falls in older adults and improve gait stability. Programs incorporating tai chi, single-leg stance, and dynamic balance challenges can decrease fall risk and related injuries when practiced regularly.",
        "[20] Watching fitness influencers online without engaging in actual physical activity does not confer physiological benefits. Passive consumption of fitness content is not a substitute for regular exercise and yields no direct cardiometabolic improvements.",
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
    )

    # 用户消息先给出query和passages，再给出指令（确保模型知道上下文后再按指令推理）
    n = len(documents)
    user_message = (
        f"Query: \"{query}\"\n\n"
        f"Passages:\n{doc_text}\n\n"
        f"{instruction}\n\n"
        f"There are {n} passages with identifiers [1]..[{n}]. In <answer>, you MUST include ALL of them exactly once (no omissions, no duplicates).\n"
        f"Now please produce the final ranking."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def load_qwen_model():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
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


def extract_answer_text_robust(response_text: str) -> str:
    """Robustly extract answer content for reward usage.
    - Prefer <answer>...</answer> (case-insensitive)
    - Else, find the longest chain like [i] > [j] > ... (at least 3 items)
    - Fallback to empty string
    """
    m = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    chain_pattern = r"(?:\[\d+\]\s*(?:>\s*\[\d+\]\s*){2,})"
    chains = re.findall(chain_pattern, response_text, flags=re.DOTALL)
    if chains:
        def id_count(s: str) -> int: return len(re.findall(r"\[\d+\]", s))
        max_count = max(id_count(s) for s in chains)
        candidates = [s for s in chains if id_count(s) == max_count]
        return candidates[-1].strip()
    return ""


def normalize_answer_chain(answer_text: str, n: int) -> tuple[list[int], str]:
    """Dedup keep-first, filter out-of-range, append missing ascending; return indices and chain string."""
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", answer_text)]
    ids = [i for i in ids if 1 <= i <= n]
    seen, dedup = set(), []
    for i in ids:
        if i not in seen:
            dedup.append(i); seen.add(i)
    missing = [i for i in range(1, n+1) if i not in seen]
    order = dedup + missing
    chain = " > ".join([f"[{i}]" for i in order])
    return order[:n], chain










def main():
    print("=== Qwen2.5-7B 灵活排序 Prompt 测试（20文档+NDCG） ===\n")
    # 1) 准备数据
    query, documents = create_test_data()
    print(f"Query: {query}")
    print("Documents:")
    for d in documents:
        print("  " + d)

    # 本测试仅输出从 <answer> 提取并清洗后的排序

    # 2) 构建消息（使用与produce_data相同的指令文本）
    messages = build_chat_messages(query, documents)

    # 3) 加载模型并生成
    try:
        model, tokenizer = load_qwen_model()
        response = generate_response(model, tokenizer, messages)
        print(response)
        print("\n\n\n")
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        print("请确保已安装 transformers、torch，并有足够内存/GPU。")
        return

    # 4) 仅提取并输出排序

    # 5) 规范化 answer 并构造 item，然后计算 reward
    #    compare_rounds_reward 要求 answer 长度与重复严格（ANSWER_EXPECTED_N=20），因此先尽力规范化链路
    extracted = extract_answer_text_robust(response)
    order, chain = normalize_answer_chain(extracted, n=len(documents))
    print("\nExtracted ranking:", chain)

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()


import json
import argparse
from typing import List, Tuple

from tqdm import tqdm

from deepteam_single.attacks.single_turn import Base64
from deepteam_single.attacks.single_turn import Multilingual
from deepteam_single.attacks.single_turn import GrayBox
from deepteam_single.attacks.single_turn import Leetspeak
from deepteam_single.attacks.single_turn import MathProblem
from deepteam_single.attacks.single_turn import PromptInjection
from deepteam_single.attacks.single_turn import Roleplay
from deepteam_single.attacks.single_turn import ROT13
from deepteam_single.attacks.single_turn import InputBypass
from deepteam_single.attacks.single_turn import GoalRedirection
from deepteam_single.attacks.single_turn import ContextPoisoning
from deepteam_single.attacks.single_turn import PermissionEscalation
from deepteam_single.attacks.single_turn import LinguisticConfusion
from deepteam_single.attacks.single_turn import SystemOverride
from custllm import CustomLLM


# 14 种单轮攻击方法与是否需要 simulator_model（对齐你原来的 test.py）
ATTACKS: List[Tuple[str, object, bool]] = [
    ("Base64", Base64(), False),
    ("Multilingual", Multilingual(), True),
    ("GrayBox", GrayBox(), True),
    ("Leetspeak", Leetspeak(), False),
    ("MathProblem", MathProblem(), True),
    ("PromptInjection", PromptInjection(), False),
    ("Roleplay", Roleplay(), True),
    ("ROT13", ROT13(), False),
    ("InputBypass", InputBypass(), False),
    ("GoalRedirection", GoalRedirection(), False),
    ("ContextPoisoning", ContextPoisoning(), False),
    ("PermissionEscalation", PermissionEscalation(), False),
    ("LinguisticConfusion", LinguisticConfusion(), False),
    ("SystemOverride", SystemOverride(), False),
]


def run_generation(question: str, output_path: str | None = "deepteam.json") -> None:
    """接收一个手动输入的问题，按 14 种方法生成并打印，可选写入 JSON。"""

    question = (question or "").strip()
    if not question:
        raise ValueError("请提供非空的初始问题。")

    results: List[dict] = []

    print("==== 使用单个问题进行 14 种方法生成 ====")
    for attack_name, attack_obj, needs_model in tqdm(
        ATTACKS,
        desc="生成中",
        unit="方法",
        leave=True,
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} ({percentage:3.0f}%)",
    ):
        try:
            if needs_model:
                answer = attack_obj.enhance(question, simulator_model=CustomLLM())
            else:
                answer = attack_obj.enhance(question)
        except Exception as e:
            answer = f"[生成失败: {e}]"

        print(f"[{attack_name}] -> {answer}")
        results.append({
            "attack_type": attack_name,
            "question": question,
            "answer": answer,
        })

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"==== 已写入: {output_path} ====")


def main() -> None:
    parser = argparse.ArgumentParser(description="手动输入一个问题，按 14 种方法生成")
    parser.add_argument("--question", required=True, help="初始问题（字符串）")
    parser.add_argument(
        "--output",
        default="deepteam.json",
        help="输出 JSON 文件路径（默认 deepteam.json；留空不写入）",
    )
    args = parser.parse_args()

    # 若用户明确设置不写出，可传空字符串
    output_path = args.output if args.output.strip() else None
    run_generation(args.question, output_path)


if __name__ == "__main__":
    main()
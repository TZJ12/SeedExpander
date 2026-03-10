from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# 解析 prompt_templates.json 的路径（相对当前文件位置）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_JSON_PATH = _PROJECT_ROOT / "attack_methods" / "prompt_templates.json"


def _load_prompt_templates() -> List[Dict]:
    """加载 prompt_templates.json 中的所有模板项。

    返回列表，每个元素为包含至少以下键的字典：
    - id: 模板 ID
    - template: 模板文本
    - attack_type: 模板所属的攻击类型标识（例如 renellm）
    - placeholder_token: 占位符标识（例如 input、harmful_content、<> 等）
    """
    if not _PROMPT_JSON_PATH.exists():
        raise FileNotFoundError(f"prompt_templates.json 不存在: {_PROMPT_JSON_PATH}")

    with _PROMPT_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 文件结构为按类别分组（例如 RolePlay、ReNeLLM 等），各组下为模板项列表。
    templates: List[Dict] = []
    for group in data.values():
        # 允许组是列表或包含子键的字典，适配不同结构
        if isinstance(group, list):
            templates.extend(group)
        elif isinstance(group, dict):
            # 可能有像 {"items": [...]} 的结构
            for v in group.values():
                if isinstance(v, list):
                    templates.extend(v)
    return templates


def _safe_fill_template(template_text: str, placeholder_token: Optional[str], value: str) -> str:
    """安全替换模板中的占位符。

    - 对于非 ReNeLLM 模板，通常占位符形如 {input} 或 {harmful_content}；
    - 对于 ReNeLLM 模板，占位符通常是 <>；
    - 避免使用 str.format，以免与模板中的花括号冲突。
    """
    # 优先使用显式占位符信息
    if placeholder_token:
        if placeholder_token == "<>":
            if "<>" in template_text:
                return template_text.replace("<>", value)
        else:
            placeholder = "{" + placeholder_token + "}"
            if placeholder in template_text:
                return template_text.replace(placeholder, value)

    # 回退：兼容不统一占位符集合
    candidates = [
        "{input}",
        "{harmful_content}",
        "{question}",
        "{text}",
        "<>",
    ]
    replaced = template_text
    did_replace = False
    for c in candidates:
        if c in replaced:
            replaced = replaced.replace(c, value)
            did_replace = True
    return replaced if did_replace else template_text


def generate_all_prompts(
    input_text: str,
    exclude_ids: Optional[List[str]] = None,
    exclude_attack_types: Optional[List[str]] = None,
) -> Dict[str, str]:
    """生成除指定排除项外的全部模板提示文本。

    - 默认排除：id 为 roleplay_01，以及 attack_type 为 renellm 与 prompt_probing 的模板。
    - 返回字典：{template_id: filled_prompt}
    """
    exclude_ids = exclude_ids or []
    exclude_attack_types = exclude_attack_types or ["renellm", "prompt_probing"]

    prompts: Dict[str, str] = {}
    for item in _load_prompt_templates():
        tid = item.get("id")
        attack_type = item.get("attack_type")
        template_text = item.get("template")
        placeholder_token = item.get("placeholder_token")

        # 跳过无效项
        if not tid or not template_text:
            continue

        # 排除逻辑
        if tid in exclude_ids:
            continue
        if attack_type and attack_type.lower() in {t.lower() for t in exclude_attack_types}:
            continue

        filled = _safe_fill_template(template_text, placeholder_token, input_text)
        prompts[tid] = filled

    return prompts


def generate_all_prompts_unified(
    input_text: str,
    exclude_ids: Optional[List[str]] = None,
    exclude_attack_types: Optional[List[str]] = None,
) -> Dict[str, str]:
    """与 generate_all_prompts 等价，但对不统一占位符进行兼容替换。

    说明：实际替换策略与 _safe_fill_template 一致，能够自动识别常见占位符
    并用输入文本进行填充，无需依赖每个模板的 placeholder_token 完全一致。
    """
    exclude_ids = exclude_ids or []
    exclude_attack_types = exclude_attack_types or ["renellm", "prompt_probing"]

    prompts: Dict[str, str] = {}
    for item in _load_prompt_templates():
        tid = item.get("id")
        attack_type = item.get("attack_type")
        template_text = item.get("template")
        placeholder_token = item.get("placeholder_token")

        if not tid or not template_text:
            continue
        if tid in exclude_ids:
            continue
        if attack_type and attack_type.lower() in {t.lower() for t in exclude_attack_types}:
            continue

        filled = _safe_fill_template(template_text, placeholder_token, input_text)
        prompts[tid] = filled
    return prompts


def fill_and_print_all_prompts(
    input_text: str,
    exclude_ids: Optional[List[str]] = None,
    exclude_attack_types: Optional[List[str]] = None,
) -> None:
    """直接填充并打印所有模板的最终内容（不依赖任何模型）。

    - 默认排除：id 为 roleplay_01；attack_type 为 renellm、prompt_probing。
    - 输出格式：每条以 "[ID]" 作为标题，随后打印填充后的文本。
    """
    prompts = generate_all_prompts_unified(
        input_text,
        exclude_ids=exclude_ids,
        exclude_attack_types=exclude_attack_types,
    )
    for tid, text in prompts.items():
        print(f"[ID] {tid}")
        print(text)
        print("\n" + "=" * 60 + "\n")


def generate_prompt_by_id(template_id: str, input_text: str) -> Optional[str]:
    """按模板 ID 生成单个提示文本；若不存在则返回 None。"""
    for item in _load_prompt_templates():
        if item.get("id") == template_id:
            return _safe_fill_template(item.get("template", ""), item.get("placeholder_token"), input_text)
    return None


__all__ = [
    "generate_all_prompts",
    "generate_all_prompts_unified",
    "fill_and_print_all_prompts",
    "generate_prompt_by_id",
    "main",
]


def main():
    parser = argparse.ArgumentParser(
        description="Fill prompt templates by ID and print final content (no model)."
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Template ID to fill, or 'all' to process all non-excluded templates.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Input question text to substitute into template placeholders.",
    )
    parser.add_argument(
        "--include-prompt-probing",
        action="store_true",
        help="Include templates with attack_type=prompt_probing (excluded by default).",
    )
    parser.add_argument(
        "--include-renellm",
        action="store_true",
        help="Include templates with attack_type=renellm (excluded by default).",
    )
    parser.add_argument(
        "--include-roleplay-01",
        action="store_true",
        help="Deprecated: roleplay_01 inclusion flag is a no-op.",
    )

    args = parser.parse_args()

    exclude_ids: List[str] = []
    exclude_attack_types: List[str] = []
    if not args.include_renellm:
        exclude_attack_types.append("renellm")
    if not args.include_prompt_probing:
        exclude_attack_types.append("prompt_probing")

    template_id = args.id.strip()
    question = args.question

    if template_id.lower() == "all":
        fill_and_print_all_prompts(
            question,
            exclude_ids=exclude_ids,
            exclude_attack_types=exclude_attack_types,
        )
        return

    # 单个模板填充：若模板在排除列表中，给出提示
    items = _load_prompt_templates()
    item = next((i for i in items if i.get("id") == template_id), None)
    if not item:
        print(f"模板ID不存在: {template_id}")
        return

    attack_type = (item.get("attack_type") or "").lower()
    if template_id in set(exclude_ids) or attack_type in set(map(str.lower, exclude_attack_types)):
        print(f"模板已在排除列表中，未填充: id={template_id}, attack_type={attack_type}")
        return

    filled = _safe_fill_template(item.get("template", ""), item.get("placeholder_token"), question)
    print(f"[ID] {template_id}")
    print(filled)


if __name__ == "__main__":
    main()

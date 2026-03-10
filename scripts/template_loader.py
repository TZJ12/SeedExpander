from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_templates_from_file(json_path: Path) -> List[Dict[str, Any]]:
    """从指定模板文件加载所有条目，支持顶层分组为 list 或嵌套 dict(list)。"""
    if not json_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    templates: List[Dict[str, Any]] = []
    for group_value in data.values():
        if isinstance(group_value, list):
            templates.extend(group_value)
        elif isinstance(group_value, dict):
            for v in group_value.values():
                if isinstance(v, list):
                    templates.extend(v)
    return templates


def load_all_templates(
    prompt_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """加载两个模板文件，返回 (全部条目, 按文件分组)。"""
    prompt_items = load_templates_from_file(prompt_path)

    by_file = {
        prompt_path.name: prompt_items,
    }
    all_items: List[Dict[str, Any]] = []
    for items in by_file.values():
        all_items.extend(items)
    return all_items, by_file

import argparse
import json
import os
from typing import Any, Dict, List, Optional


# 默认分数边界：
# [1.0, 2.0) [2.0, 5.0) [5.0, 8.5) [8.5, 10.0]
DEFAULT_EDGES = [1.0, 2.0, 5.0, 8.5, 10.0]


def flatten_loaded(loaded: Any) -> List[Dict[str, Any]]:
    """将顶层 JSON（数组或对象）展开为字典列表。
    - 顶层为数组：仅保留其中的字典项
    - 顶层为对象：遍历其值，若为数组则追加其中的字典项；若为字典则直接追加
    """
    if isinstance(loaded, list):
        return [it for it in loaded if isinstance(it, dict)]
    if isinstance(loaded, dict):
        out: List[Dict[str, Any]] = []
        for v in loaded.values():
            if isinstance(v, list):
                out.extend([it for it in v if isinstance(it, dict)])
            elif isinstance(v, dict):
                out.append(v)
        return out
    return []


def label_for_score(score: float) -> Optional[str]:
    """根据默认边界为分数返回区间标签。最后一个区间右闭。"""
    edges = DEFAULT_EDGES
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if left <= score <= right:
                return f"[{left}, {right}]"
        else:
            if left <= score < right:
                return f"[{left}, {right})"
    return None


def parse_score(item: Dict[str, Any]) -> Optional[float]:
    """解析条目中的 score 字段为浮点数。失败返回 None。"""
    s = item.get("score")
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def filter_range(items: List[Dict[str, Any]], low: float = 5.0, high: float = 10.0) -> List[Dict[str, Any]]:
    """筛选 [low, high]（两端含）范围内的条目。"""
    out: List[Dict[str, Any]] = []
    for it in items:
        val = parse_score(it)
        if val is None:
            continue
        if low <= val <= high:
            out.append(it)
    return out


def main():
    parser = argparse.ArgumentParser(description="统计分数区间数量，并导出 5.0~10.0 区间数据到新 JSON")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径（顶层为数组或对象）")
    parser.add_argument(
        "--output",
        required=False,
        help="输出 JSON 文件路径（默认：与输入同目录，文件名追加 .score_5_to_10.json）",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    items = flatten_loaded(loaded)
    if not isinstance(items, list):
        print("输入 JSON 顶层必须为数组或对象（其值为数组/对象）。")
        return

    labels = [
        "[1.0, 2.0)",
        "[2.0, 5.0)",
        "[5.0, 8.5)",
        "[8.5, 10.0]",
    ]
    counts: Dict[str, int] = {lb: 0 for lb in labels}

    for it in items:
        v = parse_score(it)
        if v is None:
            continue
        lb = label_for_score(v)
        if lb:
            counts[lb] += 1

    print("分数区间统计：")
    print(f"  [1.0,2.0): {counts['[1.0, 2.0)']} 条")
    print(f"  [2.0,5.0): {counts['[2.0, 5.0)']} 条")
    print(f"  [5.0,8.5): {counts['[5.0, 8.5)']} 条")
    print(f"  [8.5,10.0]: {counts['[8.5, 10.0]']} 条")

    filtered = filter_range(items)
    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(os.path.basename(args.input))
        out_path = os.path.join(os.path.dirname(args.input), f"{base}.score_5_to_10.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"5.0~10.0 区间数据已写入：{out_path}")


if __name__ == "__main__":
    main()
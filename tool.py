import re
import json
import ast
from typing import List, Dict, Any, Optional


def extract_instructions(text: str) -> List[str]:
    """
    从原始输入中提取所有指令内容（去掉格式）
    统一的指令提取函数，支持多种格式变体
    """
    if not text or not isinstance(text, str):
        return []
    
    # 尝试多种模式匹配，确保兼容性
    patterns = [
        r'## instruction：\s*(.*?)\s*(?=## instruction：|### response：|$)',
        r"## instruction：\s*(.*?)(?=\n## instruction：|\Z)",
        r'## instruction：\s*(.*?)(?=\n##|\Z)',
    ]
    
    instructions = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            instructions = [instr.strip() for instr in matches if instr.strip()]
            break
    
    return instructions


def rebuild_output(processed_texts: List[str]) -> str:
    """
    将处理后的内容重新拼接成原格式
    统一的输出重组函数
    """
    if not processed_texts:
        return "### response：\n"
    
    result = ["### response："]
    for text in processed_texts:
        if text and text.strip():
            result.append(f"## instruction：\n{text.strip()}\n")
    
    return "\n".join(result)


def remove_think_block(text: str) -> str:
    """
    移除文本中的思考块标签
    """
    if not text:
        return text
        
    if '</think>' in text and '<think>' not in text:
        text = '<think>' + text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def clean_llm_output(text: str) -> str:
    """
    清洗LLM输出，移除常见的格式问题
    """
    if not text:
        return ""
    
    # 移除思考块
    text = remove_think_block(text)
    
    # 移除多余的空白字符
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """
    安全解析JSON，支持多种格式和容错
    """
    if not text:
        return None
    
    # 清理文本
    text = clean_llm_output(text)
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取JSON块
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # 尝试使用ast.literal_eval
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    return None


def extract_content_by_key(text: str, key: str = "input") -> Optional[str]:
    """
    从文本中提取指定键的内容，支持JSON和其他格式
    """
    # 尝试JSON解析
    parsed = parse_json_safely(text)
    if parsed and key in parsed:
        return str(parsed[key])
    
    # 尝试正则表达式提取
    patterns = [
        rf'"{key}":\s*"([^"]*)"',
        rf"'{key}':\s*'([^']*)'",
        rf'{key}:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


def validate_and_fallback(processed_text: str, original_text: str, operation_name: str = "处理") -> str:
    """
    验证处理结果，失败时回退到原文并记录
    """
    if not processed_text or not processed_text.strip():
        print(f"警告: {operation_name}失败，保留原文")
        return original_text
    
    # 基本有效性检查
    if len(processed_text.strip()) < 3:
        print(f"警告: {operation_name}结果过短，保留原文")
        return original_text
    
    return processed_text.strip()
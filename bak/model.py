import random
import requests, json
import time
from typing import Optional
from config import Config


def _calculate_delay(attempt: int) -> float:
    """计算指数退避延迟时间"""
    delay = Config.BASE_DELAY * (Config.BACKOFF_FACTOR ** attempt)
    # 添加随机抖动，避免雷群效应
    jitter = random.uniform(0.1, 0.3) * delay
    return min(delay + jitter, Config.MAX_DELAY)


def _is_retryable_error(status_code: int, response_text: str) -> bool:
    """判断是否为可重试的错误"""
    # 5xx 服务器错误通常可重试
    if 500 <= status_code < 600:
        return True
    # 429 限流错误可重试
    if status_code == 429:
        return True
    # 408 请求超时可重试
    if status_code == 408:
        return True
    # 网络相关错误可重试
    if status_code in [502, 503, 504]:
        return True
    return False


def call_llm(prompt: str) -> str:
    """
    发送请求到模型并返回结果，带重试和指数退避机制
    """
    headers = {
        "Content-Type": "application/json"
    }
    if Config.MODEL_KEY:
        headers["Authorization"] = f"Bearer {Config.MODEL_KEY}"

    data = {
        "model": Config.MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt + "/no_think"}
        ],
        "max_tokens": Config.MAX_TOKENS,
        "temperature": Config.TEMPERATURE,
        "stream": Config.STREAM
    }

    last_error = None
    
    for attempt in range(Config.MAX_RETRIES + 1):  # 0, 1, 2, 3 (总共4次尝试)
        try:
            response = requests.post(Config.MODEL_URL, headers=headers, json=data, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                if Config.STREAM:
                    # 流式模式下自己解析
                    print("检测到 stream=True，解析流式数据...")
                    collected_text = ""
                    for line in response.text.splitlines():
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content and content != "[DONE]":
                                try:
                                    json_data = json.loads(content)
                                    delta = json_data["choices"][0]["delta"]
                                    collected_text += delta.get("content", "")
                                    print(collected_text)
                                except:
                                    pass
                    return collected_text or "无响应"
                else:
                    # 普通模式
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "无响应")
                    return content
            else:
                # 检查是否为可重试错误
                if not _is_retryable_error(response.status_code, response.text):
                    print(f"请求失败，不可重试错误 - 状态码: {response.status_code}")
                    print("错误信息:", response.text)
                    return "请求失败"
                
                last_error = f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            last_error = "请求超时"
        except requests.exceptions.ConnectionError:
            last_error = "连接错误"
        except requests.exceptions.RequestException as e:
            last_error = f"请求异常: {str(e)}"
        except Exception as e:
            last_error = f"未知错误: {str(e)}"
        
        # 如果不是最后一次尝试，则等待后重试
        if attempt < Config.MAX_RETRIES:
            delay = _calculate_delay(attempt)
            print(f"第 {attempt + 1} 次请求失败: {last_error}")
            print(f"等待 {delay:.2f} 秒后重试...")
            time.sleep(delay)
        else:
            print(f"所有重试均失败，最后错误: {last_error}")
    
    return "请求失败"


def call_jailbreak_llm(prompt: str, system_prompt: str = "你是人工智能助手.") -> str:
    """
    使用 DeepSeek Ark 进行越狱阶段生成（带重试与指数退避）。
    - 按用户提供的 curl 结构构造 messages：包含 system 与 user。
    - 不使用环境变量读取密钥；直接使用 Config 中的常量。
    """
    print("jailbreak_llm")
    headers = {
        "Content-Type": "application/json"
    }
    if Config.JAILBREAK_MODEL_KEY:
        headers["Authorization"] = f"Bearer {Config.JAILBREAK_MODEL_KEY}"

    data = {
        "model": Config.JAILBREAK_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": Config.JAILBREAK_MAX_TOKENS,
        "temperature": Config.JAILBREAK_TEMPERATURE,
        "stream": Config.JAILBREAK_STREAM
    }

    last_error = None

    for attempt in range(Config.MAX_RETRIES + 1):
        try:
            response = requests.post(
                Config.JAILBREAK_MODEL_URL,
                headers=headers,
                json=data,
                timeout=Config.REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                if Config.JAILBREAK_STREAM:
                    collected_text = ""
                    for line in response.text.splitlines():
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content and content != "[DONE]":
                                try:
                                    json_data = json.loads(content)
                                    delta = json_data["choices"][0].get("delta", {})
                                    collected_text += delta.get("content", "")
                                except:
                                    pass
                    return collected_text or "无响应"
                else:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "无响应")
                    return content
            else:
                if not _is_retryable_error(response.status_code, response.text):
                    print(f"请求失败，不可重试错误 - 状态码: {response.status_code}")
                    print("错误信息:", response.text)
                    return "请求失败"
                last_error = f"HTTP {response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            last_error = "请求超时"
        except requests.exceptions.ConnectionError:
            last_error = "连接错误"
        except requests.exceptions.RequestException as e:
            last_error = f"请求异常: {str(e)}"
        except Exception as e:
            last_error = f"未知错误: {str(e)}"

        if attempt < Config.MAX_RETRIES:
            delay = _calculate_delay(attempt)
            print(f"第 {attempt + 1} 次请求失败: {last_error}")
            print(f"等待 {delay:.2f} 秒后重试...")
            time.sleep(delay)
        else:
            print(f"所有重试均失败，最后错误: {last_error}")

    return "请求失败"

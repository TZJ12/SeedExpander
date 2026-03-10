import random, httpx
import asyncio
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

_async_client = httpx.AsyncClient(
    timeout=Config.REQUEST_TIMEOUT,
    limits=httpx.Limits(
        max_connections=200,
        max_keepalive_connections=50
    )
)

async def call_llm_async(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if Config.MODEL_KEY:
        headers["Authorization"] = f"Bearer {Config.MODEL_KEY}"

    data = {
        "model": Config.MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": Config.MAX_TOKENS,
        "temperature": Config.TEMPERATURE,
        "stream": False,   # ✅ 强制关闭
    }

    last_error = None

    for attempt in range(Config.MAX_RETRIES + 1):
        try:
            r = await _async_client.post(
                Config.MODEL_URL,
                headers=headers,
                json=data,
            )

            if r.status_code != 200:
                if not _is_retryable_error(r.status_code, r.text):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                last_error = r.text
                raise RuntimeError(last_error)

            result = r.json()
            content = result["choices"][0]["message"]["content"]

            # ✅ JSON完整性快速检查（给deepeval用）
            if content.count("{") and not content.strip().endswith("}"):
                raise RuntimeError("LLM output truncated")

            return content

        except Exception as e:
            last_error = str(e)

            if attempt >= Config.MAX_RETRIES:
                raise RuntimeError(f"LLM failed after retries: {last_error}")

            delay = _calculate_delay(attempt)
            await asyncio.sleep(delay)



async def call_jailbreak_llm_async(
    prompt: str,
    system_prompt: str = "你是人工智能助手."
) -> str:

    headers = {"Content-Type": "application/json"}

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
        "stream": False,   # ✅ 强制关闭stream
    }

    last_error = None

    for attempt in range(Config.MAX_RETRIES + 1):
        try:
            r = await _async_client.post(
                Config.JAILBREAK_MODEL_URL,
                headers=headers,
                json=data,
            )

            if r.status_code != 200:
                if not _is_retryable_error(r.status_code, r.text):
                    raise RuntimeError(
                        f"HTTP {r.status_code}: {r.text}"
                    )
                last_error = r.text
                raise RuntimeError(last_error)

            result = r.json()

            content = result["choices"][0]["message"]["content"]

            if not content or not content.strip():
                raise RuntimeError("LLM returned empty content")

            # ✅ JSON完整性检测（给deepeval链路用）
            if content.count("{") and not content.strip().endswith("}"):
                raise RuntimeError("LLM output appears truncated")

            return content

        except Exception as e:
            last_error = str(e)

            if attempt >= Config.MAX_RETRIES:
                raise RuntimeError(
                    f"Jailbreak LLM failed after retries: {last_error}"
                )

            delay = _calculate_delay(attempt)
            print(f"Jailbreak retry {attempt+1}: {last_error}")
            await asyncio.sleep(delay)



import asyncio
import functools
import json
from socket import timeout
import httpx
import traceback
from deepeval.models import DeepEvalBaseLLM
from config import Config


class CustomLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.api_url = Config.MODEL_URL
        self.api_key = Config.MODEL_KEY
        self.model_name = Config.MODEL_NAME
        self.max_tokens = 4096
        self.temperature = 0.7
        self.stream = False  # 默认不开流式，可按需改 True
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(180),
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=100
            ))

    def get_model_name(self):
        return self.model_name

    def load_model(self):
        return self

    async def generate(self, prompt: str, schema: dict = None, **kwargs) -> str:
        """
        异步生成函数，使用 httpx 发送请求
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream
        }

        try:

            # ========= 普通模式 =========
            if not self.stream:
                try:
                    resp = await self.client.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=180
                    )
                except httpx.ReadTimeout as e:
                    print("[CustomLLM] ReadTimeout:", str(e))
                    print(traceback.format_exc())
                    return {
                        "input": "",
                        "error": "ReadTimeout",
                        "message": str(e)
                    }

                if resp.status_code != 200:
                    print("[CustomLLM] HTTP ERROR:", resp.status_code)
                    print(resp.text[:500])
                    return self._fallback_json()

                result = resp.json()

                raw_result = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            # ========= 流式模式 =========
            else:
                raw_result = ""
                async with self.client.stream(
                        "POST",
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=180
                ) as resp:

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content == "[DONE]":
                                break

                            try:
                                chunk = json.loads(content)
                                delta = chunk["choices"][0]["delta"]
                                raw_result += delta.get("content", "")
                            except:
                                pass

        except Exception as e:
            print(f"\n[CustomLLM] Request Exception:{e}")
            traceback.print_exc()
            return self._fallback_json()

        fixed = self._extract_json(raw_result)

        return fixed

    # =========================
    # JSON 提取
    # =========================
    def _extract_json(self, text: str) -> str:
        if "{" in text and "}" in text:
            try:
                json.loads(text)
                return text
            except:
                start = text.find("{")
                end = text.rfind("}") + 1
                return text[start:end]

        return text

    # =========================
    # DeepEval schema 兜底
    # =========================
    def _fallback_json(self):
        return json.dumps({
            "non_compliant": False,
            "reason": "llm_request_failed"
        })

    # =========================
    # deepeval async 接口
    # =========================
    async def a_generate(self, prompt: str, schema: dict = None, **kwargs) -> str:
        return await self.generate(prompt, schema=schema, **kwargs)


# =========================
# 测试
# =========================
async def main():
    llm = CustomLLM()
    result = await llm.a_generate("你叫啥")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())


import requests
import json
import asyncio, functools
from deepeval.models import DeepEvalBaseLLM
from config import Config

class CustomLLM(DeepEvalBaseLLM):
    def __init__(self):
        # 模型配置
        self.api_url = Config.MODEL_URL
        self.api_key = Config.MODEL_KEY
        self.model_name = Config.MODEL_NAME
        # 降低补全长度，避免超过上下文窗口
        self.max_tokens = 4096
        # 降低随机性，提升格式稳定性
        self.temperature = 0.7
        self.stream = False

    def get_model_name(self):
        return self.model_name

    def load_model(self):
        return self

    def generate(self, prompt: str, schema: dict = None, **kwargs) -> str:
        """
        发送请求到模型并返回结果
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            # 如果服务兼容 OpenAI 格式，强制输出 JSON
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        except Exception as e:
            print(f"[CustomLLM] Request failed: {e}")
            return '{"error": "Request failed"}'

        if response.status_code == 200:
            if self.stream:
                # 流式模式下自己解析
                collected_text = ""
                for line in response.text.splitlines():
                    if line.startswith("data: "):
                        content = line[6:].strip()
                        if content and content != "[DONE]":
                            try:
                                json_data = json.loads(content)
                                delta = json_data["choices"][0]["delta"]
                                collected_text += delta.get("content", "")
                            except:
                                pass
                raw_result = collected_text or "无响应"
            else:
                # 普通模式
                try:
                    result = response.json()
                    raw_result = result.get("choices", [{}])[0].get("message", {}).get("content", "无响应")
                except Exception:
                    raw_result = response.text
            
            # 尝试修复/提取 JSON
            # 很多时候模型会输出 ```json {...} ``` 或者包含解释性文字
            # 我们尝试找到第一个 { 和最后一个 }
            if "{" in raw_result and "}" in raw_result:
                try:
                    # 快速验证是否已经是合法 JSON
                    json.loads(raw_result)
                    return raw_result
                except:
                    # 提取 { ... }
                    start = raw_result.find("{")
                    end = raw_result.rfind("}") + 1
                    potential_json = raw_result[start:end]
                    return potential_json
            
            return raw_result

        else:
            print("请求失败，错误信息:", response.text)
            return "请求失败"

    async def a_generate(self, prompt: str, schema: dict = None, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        # schema and kwargs will be passed to self.generate
        return await loop.run_in_executor(None, functools.partial(self.generate, prompt, schema=schema, **kwargs))

if __name__ == "__main__":
    cust_llm = CustomLLM()
    print(cust_llm.generate("给我讲一个笑话"))


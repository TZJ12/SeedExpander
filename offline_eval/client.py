import sys
from pathlib import Path
import asyncio
import time
from typing import Dict, Any, List, Optional
from collections import deque
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 将项目根目录添加到Python路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 获取项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = deque()
        self.token_usage = deque()
        self.request_lock = asyncio.Lock()
        
    async def wait_if_needed(self, tokens: int) -> None:
        async with self.request_lock:
            now = time.time()
            minute_ago = now - 60
            
            # 清理旧的记录
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            while self.token_usage and self.token_usage[0][0] < minute_ago:
                self.token_usage.popleft()
            
            # 检查请求频率
            while len(self.request_times) >= self.requests_per_minute:
                sleep_time = self.request_times[0] - minute_ago
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.time()
                minute_ago = now - 60
                while self.request_times and self.request_times[0] < minute_ago:
                    self.request_times.popleft()
            
            # 检查token使用量
            current_tokens = sum(t for _, t in self.token_usage)
            while current_tokens + tokens > self.tokens_per_minute:
                sleep_time = self.token_usage[0][0] - minute_ago
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.time()
                minute_ago = now - 60
                while self.token_usage and self.token_usage[0][0] < minute_ago:
                    self.token_usage.popleft()
                current_tokens = sum(t for _, t in self.token_usage)
            
            # 记录新的请求
            self.request_times.append(now)
            self.token_usage.append((now, tokens))

class ApiClient:
    def __init__(
            self, 
            api_key: str,
            base_url: str = None, # type: ignore
            model: str = None, # type: ignore
            max_concurrent: int = 5,
            requests_per_minute: int = 500,
            tokens_per_minute: int = 90000,
            timeout: float = 30.0,
            enable_rate_limit: bool = False,  # 新增参数
            max_retries: int = 0
        ):
        self.api_key = api_key
        self.base_url = base_url 
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model = model
        self.enable_rate_limit = enable_rate_limit
        self.max_retries = max_retries
        
        # 只有启用限制时才创建rate_limiter
        if self.enable_rate_limit:
            self.rate_limiter = RateLimiter(requests_per_minute, tokens_per_minute)
        else:
            self.rate_limiter = None
        
        self.timeout = timeout
        
        # 初始化AsyncOpenAI客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self.max_retries  # 禁用自动重试，由我们自己控制
        )
        
    def count_tokens(self, messages: List[Dict]) -> int:
        """简单估算token数量"""
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        return int(total_chars * 0.5)
    
    async def chat_completion(
        self, 
        messages: list, 
        model: str = None, # type: ignore
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stream: bool = False,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 30,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        response_format: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        estimated_tokens = self.count_tokens(messages)
        
        # 只有启用限制时才等待
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed(estimated_tokens) # type: ignore
        
        if not model:
            model = self.model
            
        async with self.semaphore:
            try:
                # 构建基础参数
                create_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                }
                
                # 根据模型类型添加max_tokens参数
                if model == 'o3-mini':
                    create_kwargs["max_completion_tokens"] = max_tokens
                else:
                    create_kwargs["max_tokens"] = max_tokens
                
                # 添加可选参数
                if stop:
                    create_kwargs["stop"] = stop
                    
                # 只有在提供response_format时才添加
                if response_format:
                    if response_format in ['text', 'json_object', 'json_schema']:
                        create_kwargs["response_format"] = {"type": response_format}
                
                # 添加额外参数（如果API支持）
                extra_body = {}
                if min_p != 0.0:
                    extra_body["min_p"] = min_p
                if top_k != 30:
                    extra_body["top_k"] = top_k
                if repetition_penalty != 1.0:
                    extra_body["repetition_penalty"] = repetition_penalty
                if extra_body:
                    create_kwargs["extra_body"] = extra_body
                
                # 使用AsyncOpenAI调用API
                response = await self.client.chat.completions.create(**create_kwargs)
                
                # 提取结果
                actual_tokens = response.usage.total_tokens if response.usage else estimated_tokens
                
                # 获取content
                message = response.choices[0].message
                if response_format in ["json_object", "json_schema"]:
                    # 优先取content，如果没有则尝试reasoning_content
                    content = message.content if message.content else getattr(message, 'reasoning_content', None)
                else:
                    content = message.content
                
                result = {
                    'content': content,
                    'tokens': actual_tokens,
                }
                return result
                
            except asyncio.TimeoutError as e:
                print(f"请求超时: {str(e)}")
                return {
                    'content': None,
                    'tokens': estimated_tokens,
                    'error': '请求超时'
                }
            except Exception as e:
                print(f"API请求失败: {str(e)}")
                return {
                    'content': None,
                    'tokens': estimated_tokens,
                    'error': str(e)
                }

    async def process_messages(self, model_name, messages_list: list, max_tokens: int = 1024, temperature: float = 0.8, top_p: float = 1.0, response_format: str = None, top_k: int = 30, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repetition_penalty: float = 1.0, stop: Optional[List[str]] = None) -> list: # type: ignore
        """
        并发处理多个消息列表
        """
        tasks = [self.chat_completion(messages, model=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, response_format=response_format, top_k=top_k, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, repetition_penalty=repetition_penalty, stop=stop) for messages in messages_list]
        responses = await tqdm.gather(*tasks, desc="处理消息中")
        return responses
    
    async def process_messages_stream(
            self,
            model_name, 
            messages_list: list, 
            max_tokens: int = 1024, 
            temperature: float = 0.8, 
            top_p: float = 1.0, 
            response_format: Optional[str] = None, 
            top_k: int = 30, 
            min_p: float = 0.0,
            ): # type: ignore
        """
        流式处理多个消息列表，每完成一个就立即返回
        """
        total_tokens = 0
        
        # 创建所有任务
        tasks = []
        for i, messages in enumerate(messages_list):
            task = asyncio.create_task(
                self.chat_completion(messages, model=model_name, max_tokens=max_tokens, top_k=top_k, min_p=min_p,
                                   temperature=temperature, top_p=top_p, response_format=response_format)
            )
            tasks.append((i, task))
        
        with tqdm(total=len(tasks), desc="处理消息中") as pbar:
            # 等待任务完成并立即处理
            pending = {task: idx for idx, task in tasks}
            
            while pending:
                # 等待至少一个任务完成
                done, still_pending = await asyncio.wait(
                    pending.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 处理完成的任务
                for completed_task in done:
                    try:
                        response = await completed_task
                        original_index = pending[completed_task]
                        
                        total_tokens += response.get('tokens', 0)
                        pbar.update(1)
                        
                        # 立即返回结果
                        yield (original_index, response)
                        
                    except Exception as e:
                        original_index = pending[completed_task]
                        print(f"处理消息 {original_index} 时发生错误: {str(e)}")
                        yield (original_index, {'error': str(e), 'tokens': 0})
                    
                    # 从待处理列表中移除
                    del pending[completed_task]
        
        print(f"\n总计使用token数: {total_tokens}")

import asyncio
import sys
import os
import json
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm
import aiofiles

# 添加项目根目录到Python路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from offline_eval.client import ApiClient

# ==================== 模型接入配置 ====================
DEFAULT_API_URL = "http://58.214.239.10:18080/app-2512170147-llm/v1"
DEFAULT_API_KEY = "sk-390feb24-daee-11f0-a581-0242ac150003"
DEFAULT_MODEL_NAME = "Qwen3-30B"


def read_json_file(file_path: str):
    """读取JSON文件"""
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise ValueError("文件路径不存在或不是一个有效的文件")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_json_file(file_path: str, data: list):
    """写入JSON文件"""
    dir_path = os.path.dirname(file_path)
    if dir_path:  # 只有当目录路径不为空时才创建目录
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='批量生成模型响应')
    parser.add_argument('--input', type=str, required=True,
                        help='输入的JSON文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出的JSON文件路径')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                        help=f'API key，默认为 {DEFAULT_API_KEY}')
    parser.add_argument('--base_url', type=str, default=DEFAULT_API_URL,
                        help=f'Base URL，默认为 {DEFAULT_API_URL}')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help=f'Model name（将作为响应字段名的一部分），默认为 {DEFAULT_MODEL_NAME}')
    parser.add_argument('--max_concurrent', type=int, default=10,
                        help='最大并发数，默认为10')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='超时时间，默认为60.0')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='最大token数，默认为1024')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='温度参数，默认为0.0')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始索引，默认为0')
    parser.add_argument('--end_idx', type=int, default=10,
                        help='结束索引，默认处理所有')
    parser.add_argument('--enable_rate_limit', action='store_true',
                        help='是否启用速率限制')
    parser.add_argument('--requests_per_minute', type=int, default=500,
                        help='每分钟最大请求数，默认500')
    parser.add_argument('--tokens_per_minute', type=int, default=90000,
                        help='每分钟最大token数，默认90000')

    args = parser.parse_args()

    # 确保模型名称不以斜杠结尾
    if args.model_name.endswith('/'):
        args.model_name = args.model_name.rstrip('/')

    return args


async def test_connection(client: ApiClient, model_name: str):
    """测试连接"""
    print("=== 测试连接 ===")

    test_prompt = "你好，请问今天天气如何？"
    test_messages = [{"role": "user", "content": test_prompt}]

    try:
        response = await client.chat_completion(
            messages=test_messages,
            model=model_name,
            max_tokens=100,
            temperature=0.0
        )

        if response.get('error'):
            print(f"连接测试失败: {response['error']}")
            return False
        else:
            print(f"连接测试成功!")
            print(f"测试提示: {test_prompt}")
            print(f"模型响应: {response['content'][:100]}...")
            return True
    except Exception as e:
        print(f"连接测试异常: {str(e)}")
        return False


async def main():
    args = parse_args()

    print("批量生成模型响应工具")
    print("=" * 60)
    print(f"连接到: {args.base_url}")
    print(f"模型: {args.model_name}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"最大并发: {args.max_concurrent}")
    print(f"速率限制: {'启用' if args.enable_rate_limit else '禁用'}")

    # 创建客户端
    client = ApiClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model_name,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        enable_rate_limit=args.enable_rate_limit,
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute
    )

    # 测试连接
    connection_ok = await test_connection(client, args.model_name)
    if not connection_ok:
        print("连接失败，请检查API服务")
        return

    # 读取数据
    print(f"\n读取数据文件: {args.input}")
    try:
        data = read_json_file(args.input)
        print(f"成功读取数据: {len(data)} 条")

        # 验证数据格式
        if not isinstance(data, list):
            raise ValueError("输入文件必须是一个JSON数组")

        if not data:
            print("数据为空，无需处理")
            return

        # 检查第一条数据是否有prompt字段
        if 'prompt' not in data[0]:
            raise ValueError("数据格式错误：缺少'prompt'字段")

    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 过滤数据范围
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    filtered_data = data[args.start_idx:end_idx]

    print(f"处理范围: {args.start_idx} - {end_idx}")
    print(f"实际处理: {len(filtered_data)} 条记录")

    if not filtered_data:
        print("没有数据需要处理")
        return

    # 准备临时文件和结果存储
    temp_file = args.output + '.temp.jsonl'
    response_field_name = f"{args.model_name.replace('/', '_')}_response"

    # 检查是否有临时文件（用于断点续传）
    processed_indices = set()
    partial_results = {}

    if os.path.exists(temp_file):
        print("发现临时文件，加载已处理的结果...")
        try:
            async with aiofiles.open(temp_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        result = json.loads(line.strip())
                        idx = result.get('original_idx')
                        if idx is not None:
                            processed_indices.add(idx)
                            partial_results[idx] = result.get('response', '')
                    except:
                        continue
            print(f"已处理 {len(processed_indices)} 条记录")
        except Exception as e:
            print(f"读取临时文件失败: {e}")

    # 过滤未处理的数据
    unprocessed_data = []
    for i, item in enumerate(filtered_data):
        original_idx = args.start_idx + i
        if original_idx not in processed_indices:
            unprocessed_data.append((original_idx, item))

    print(f"待处理: {len(unprocessed_data)} 条记录")

    if not unprocessed_data:
        print("所有数据已处理完成，开始合并结果...")
    else:
        try:
            print(f"\n开始生成响应...")

            # 定义处理单个item的函数
            async def process_single_item(original_idx: int, item: dict):
                try:
                    prompt = item['prompt']
                    messages = [{"role": "user", "content": prompt}]

                    response = await client.chat_completion(
                        messages=messages,
                        model=args.model_name,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature
                    )

                    if response.get('error'):
                        print(f"\n处理索引 {original_idx} 时出错: {response['error']}")
                        content = f"[ERROR] {response['error']}"
                    else:
                        content = response.get('content', '')
                        if not content:
                            print(f"\n索引 {original_idx} 返回内容为空")
                            content = "[EMPTY RESPONSE]"

                    # 保存到临时文件
                    temp_result = {
                        'original_idx': original_idx,
                        'response': content
                    }

                    async with aiofiles.open(temp_file, 'a', encoding='utf-8') as f:
                        await f.write(json.dumps(temp_result, ensure_ascii=False) + '\n')

                    return (original_idx, content)

                except Exception as e:
                    print(f"\n处理索引 {original_idx} 时发生异常: {e}")
                    error_content = f"[EXCEPTION] {str(e)}"

                    # 即使出错也保存
                    temp_result = {
                        'original_idx': original_idx,
                        'response': error_content
                    }
                    async with aiofiles.open(temp_file, 'a', encoding='utf-8') as f:
                        await f.write(json.dumps(temp_result, ensure_ascii=False) + '\n')

                    return (original_idx, error_content)

            # 使用流式处理，实时显示进度
            completed_count = 0
            error_count = 0

            # 创建所有任务
            tasks = []
            for original_idx, item in unprocessed_data:
                task = asyncio.create_task(process_single_item(original_idx, item))
                tasks.append(task)

            # 使用tqdm显示进度
            with tqdm(total=len(tasks), desc="生成响应") as pbar:
                for coro in asyncio.as_completed(tasks):
                    original_idx, content = await coro
                    partial_results[original_idx] = content

                    completed_count += 1
                    if content.startswith('[ERROR]') or content.startswith('[EXCEPTION]'):
                        error_count += 1

                    pbar.update(1)

                    # 每10条显示一次统计
                    if completed_count % 10 == 0:
                        pbar.set_postfix({
                            'completed': completed_count,
                            'errors': error_count
                        })

            print(f"\n处理完成: {completed_count} 条, 错误: {error_count} 条")

            # 自动重试失败的记录
            max_retry_rounds = 3
            retry_round = 0

            while error_count > 0 and retry_round < max_retry_rounds:
                retry_round += 1
                print(f"\n{'=' * 60}")
                print(f"第 {retry_round} 轮重试，待重试: {error_count} 条")
                print(f"{'=' * 60}")

                # 收集失败的记录
                failed_items = []
                for original_idx, content in partial_results.items():
                    if isinstance(content, str) and (
                            content.startswith('[ERROR]') or content.startswith('[EXCEPTION]')):
                        # 找到对应的原始数据
                        for idx, item in unprocessed_data:
                            if idx == original_idx:
                                failed_items.append((original_idx, item))
                                break

                if not failed_items:
                    break

                # 重试失败的记录
                retry_tasks = []
                for original_idx, item in failed_items:
                    task = asyncio.create_task(process_single_item(original_idx, item))
                    retry_tasks.append(task)

                retry_error_count = 0
                retry_success_count = 0

                # 使用tqdm显示重试进度
                with tqdm(total=len(retry_tasks), desc=f"重试轮次 {retry_round}") as pbar:
                    for coro in asyncio.as_completed(retry_tasks):
                        original_idx, content = await coro
                        partial_results[original_idx] = content

                        if content.startswith('[ERROR]') or content.startswith('[EXCEPTION]'):
                            retry_error_count += 1
                        else:
                            retry_success_count += 1

                        pbar.update(1)
                        pbar.set_postfix({
                            'success': retry_success_count,
                            'failed': retry_error_count
                        })

                print(f"重试结果: 成功 {retry_success_count} 条, 仍失败 {retry_error_count} 条")
                error_count = retry_error_count

                if retry_error_count == 0:
                    print(f"✅ 所有失败记录已重试成功！")
                    break

            if error_count > 0:
                print(f"\n⚠️  经过 {retry_round} 轮重试后，仍有 {error_count} 条记录失败")

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")
            print(f"临时文件保存在: {temp_file}")
            print("可以重新运行命令继续处理未完成的数据")
            return

    # 合并结果到原始数据
    print("\n合并结果到原始数据...")
    final_results = []

    for i, item in enumerate(filtered_data):
        original_idx = args.start_idx + i
        # 创建新的dict，包含原始数据的所有字段
        result_item = item.copy()
        # 添加响应字段
        result_item[response_field_name] = partial_results.get(original_idx, '[NOT PROCESSED]')
        final_results.append(result_item)

    # 保存最终结果
    try:
        print(f"\n保存结果到: {args.output}")
        write_json_file(args.output, final_results)

        # 删除临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("已删除临时文件")

        # 统计
        total_processed = len([r for r in final_results if not r[response_field_name].startswith('[NOT PROCESSED]')])
        error_responses = len([r for r in final_results if
                               r[response_field_name].startswith('[ERROR]') or r[response_field_name].startswith(
                                   '[EXCEPTION]')])
        success_responses = total_processed - error_responses

        print(f"\n=== 完成 ===")
        print(f"总计处理: {len(final_results)} 条")
        print(f"成功生成: {success_responses} 条")
        print(f"失败: {error_responses} 条")
        print(f"响应字段名: {response_field_name}")
        print(f"结果已保存至: {args.output}")

    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        print(f"临时文件保留在: {temp_file}")
        print("可以手动合并结果")


if __name__ == "__main__":
    asyncio.run(main())

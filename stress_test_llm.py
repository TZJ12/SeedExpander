import asyncio
import time
import statistics
from custllm import CustomLLM


# =====================
# 参数区（你可以改）
# =====================

TOTAL_REQUESTS = 200          # 总请求数
CONCURRENCY = 50              # 并发数（压测强度）
PROMPT = "你是谁，用一句话回答"


# =====================
# 单请求任务
# =====================

async def one_call(llm: CustomLLM, idx: int, sem: asyncio.Semaphore, stats: list):

    async with sem:
        start = time.time()

        try:
            result = await llm.a_generate(PROMPT)

            ok = isinstance(result, str) and len(result) > 0

            cost = time.time() - start
            stats.append(cost)

            print(f"[OK] #{idx} {cost:.2f}s  len={len(result)}")

            return ok

        except Exception as e:
            cost = time.time() - start
            stats.append(cost)

            print(f"[ERR] #{idx} {cost:.2f}s {e}")
            return False


# =====================
# 主压测
# =====================

async def main():

    llm = CustomLLM()

    sem = asyncio.Semaphore(CONCURRENCY)
    stats = []

    print("\n====================")
    print("LLM 压测开始")
    print("总请求:", TOTAL_REQUESTS)
    print("并发:", CONCURRENCY)
    print("====================\n")

    start_all = time.time()

    tasks = [
        one_call(llm, i, sem, stats)
        for i in range(TOTAL_REQUESTS)
    ]

    results = await asyncio.gather(*tasks)

    total_cost = time.time() - start_all

    success = sum(results)
    fail = TOTAL_REQUESTS - success

    print("\n====================")
    print("压测完成")
    print("====================")

    print("成功:", success)
    print("失败:", fail)
    print("成功率:", f"{success/TOTAL_REQUESTS*100:.1f}%")

    print("总耗时:", f"{total_cost:.2f}s")
    print("吞吐:", f"{TOTAL_REQUESTS/total_cost:.2f} req/s")

    if stats:
        print("平均耗时:", f"{statistics.mean(stats):.2f}s")
        print("P95:", f"{statistics.quantiles(stats, n=20)[18]:.2f}s")
        print("最大:", f"{max(stats):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())


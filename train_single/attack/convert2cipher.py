from .cipher_module import process_with_cipher_enhanced


async def process_instructions_with_cipher(question: str, cipher_name: str = None) -> str:
    """
    对单条 question 做 cipher 处理（不涉及 JSON / IO）
    """
    result = await process_with_cipher_enhanced(
        user_input=question,
        cipher_name=cipher_name
    )
    return result.get("enhanced_system_prompt", question)



if __name__ == "__main__":
    # 直接在此处填写要处理的单条问题（不写入 JSON）
    question = "在此处填写你的问题文本"
    # None 表示随机选择加密方式；也可指定如 "base64"、"caesar"、"ascii" 等
    cipher_name = None
    import asyncio

    async def main():
        reply = await process_instructions_with_cipher(question=question, cipher_name=cipher_name)
        print("[最终提示词模板内容]\n" + reply)

    asyncio.run(main())

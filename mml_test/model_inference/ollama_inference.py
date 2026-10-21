from utils.api_utils import generate_from_openai_chat_completion
from utils.input_processing import *
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from openai import AsyncOpenAI
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
import asyncio

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    output_path: str,
                    n: int=1,
                    api_base: str=""):
    """
    Generate responses from Ollama model.
    """
    base_rate = 500
    
    # 设置Ollama的API配置
    api_key = "ollama"  # Ollama不需要真实的API密钥
    base_url = api_base if api_base else "http://192.168.1.100:11434/v1/"

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 准备输入消息
    messages = prepare_qa_inputs(model_name, queries, prompt=prompt)

    # 移除模型名称中的ollama/前缀
    if model_name.startswith('ollama/'):
        model_name = model_name[7:]
    
    # 生成响应
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = int(base_rate),
            n = n
        )
    )

    # 处理响应
    for query, response in zip(queries, responses):
        query["response"] = response
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存响应
    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print(f"\nSaving responses to: {output_path}")
    print(f"Number of responses: {len(responses)}")

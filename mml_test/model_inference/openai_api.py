from utils.api_utils import generate_from_openai_chat_completion, generate_from_openai_chat_completion_with_callback
from utils.input_processing import *
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from openai import AsyncOpenAI
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# import logging 
import asyncio
import re

STOP_FLAG = False
SAVE_LOCK = asyncio.Lock()
STOP_FILE = "STOP_PROGRAM"

def check_stop_signal():
    global STOP_FLAG
    if os.path.exists(STOP_FILE):
        STOP_FLAG = True
        try:
            os.remove(STOP_FILE)
        except:
            pass
        return True
    return False

async def generate_response(model_name: str,
                    prompt: str,
                    queries: list,
                    output_path: str,
                    n: int=1,
                    api_base: str=""):
    if model_name == "kimi-k2.6":
        base_rate = 200
    else:
        base_rate = 200
    max_token = MAX_TOKENS
    if model_name.startswith("qwen"): #千问。
        if(model_name=="qwen3.6-flash"): #qwen3.6-35b-a3b
            client = AsyncOpenAI(
                api_key="******",
                base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            )
        elif(model_name=="qwen3.6-plus"):
            client = AsyncOpenAI(
                api_key="********",
                base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            )
        elif(model_name=='qwen3-vl-plus'):
            client = AsyncOpenAI(
                api_key="******",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
    elif model_name.startswith("kimi-k2.6"):
        client = AsyncOpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
            api_key="******",
        )
    elif model_name.startswith("glm-5.1"):
        client = AsyncOpenAI(
            api_key="********",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif model_name.startswith("minimax-m2.7"):
        client = AsyncOpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
            api_key="****************",
        )
    elif model_name.startswith("doubao"): # 豆包。
        client = AsyncOpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
            api_key="*****",
        )
    elif model_name.startswith("google/medgemma-1.5-4b-it") or model_name.startswith("medgemma"):
        client = AsyncOpenAI(
            base_url="https://ms-ens-ed797528-b690.api-inference.modelscope.cn/v1",
            api_key="************",
        )
        max_token = 5000
        base_rate = 5
    elif model_name.startswith("Baichuan-M3-Plus"):
        client = AsyncOpenAI(
            base_url="https://api.baichuan-ai.com/v1/",
            api_key="**********",
        )
        base_rate=50
    elif model_name.startswith("gpt-5.5") or model_name.startswith("gpt-5.4"):
        client = AsyncOpenAI(
            base_url="https://claude-api.cloud/v1",
            api_key="***********",
        )
        base_rate = 50
    elif model_name.startswith("gpt-4o"):
        client = AsyncOpenAI(
            base_url="https://www.yunqiaoai.top/v1",
            api_key="******",
        )
        base_rate = 50
        max_token = 16384
    elif model_name.startswith("gemini-3.1-pro-preview"):
        client = AsyncOpenAI(
            api_key="*******",  # 替换成你的实际 key
            base_url="https://www.yunqiaoai.top/v1"  # 替换成 sub2api 提供的实际地址
        )
        base_rate = 50

        
    messages = prepare_qa_inputs(model_name, queries, prompt=prompt)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    check_stop_signal()
    if STOP_FLAG:
        print("Stop signal detected before starting, exiting...")
        return
    
    # 定义一个回调函数，每完成一个请求就保存
    async def save_on_complete(response, query_index):
        queries[query_index]["response"] = response
        queries[query_index]["model_name"] = model_name
        
        # 判断结果：支持提取 [answer] 格式，或取最后一个词
        if response and isinstance(response, str) and response.strip():
            # 先尝试匹配方括号格式，如 "[ C ]" 或 "[ yes]"
            bracket_match = re.search(r'\[\s*(\w+)\s*\]', response.strip())
            if bracket_match:
                answer = bracket_match.group(1).strip()
            else:
                # 否则取最后一个词，去除末尾标点
                last_word = response.strip().split()[-1]
                answer = re.sub(r'[^\w]+$', '', last_word)
            
            if answer.lower() == "yes":
                queries[query_index]["result"] = True
            elif answer.lower() == "no":
                queries[query_index]["result"] = False
            elif len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                queries[query_index]["result"] = answer.upper()
            else:
                queries[query_index]["result"] = "unknown"
        else:
            queries[query_index]["result"] = "none"
        
        # 保存到文件
        async with SAVE_LOCK:
            check_stop_signal()
            if STOP_FLAG:
                print("Received stop signal, saving and exiting...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(queries, f, indent=4, ensure_ascii=False)
            if STOP_FLAG:
                print(f"Saved to {output_path}, exiting.")
                os._exit(0)
            
        #print(f"Saved response {query_index+1}/{len(queries)} to {output_path}")
    
    # 异步执行并实时保存
    responses = await generate_from_openai_chat_completion_with_callback(
        client,
        messages=messages, 
        engine_name=model_name,
        max_tokens=max_token,
        temperature=GENERATION_TEMPERATURE,
        requests_per_minute = base_rate,
        n = n,
        callback=save_on_complete,
        queries=queries
    )
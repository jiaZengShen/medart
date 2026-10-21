from utils.api_utils import generate_from_claude_chat_completion_with_callback
from utils.input_processing import *
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from anthropic import AsyncAnthropic
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
import asyncio
import re

load_dotenv()

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
                            n: int = 1,
                            api_base: str = ""):
    # ── 根据模型名称设置 API Key 和 base_url ─────────────
    if "claude" in model_name:
        client = AsyncAnthropic(
            api_key="*******",
            base_url="https://claude-api.cloud"
        )
        base_rate = 10  # 请求间隔 ~6s，防止中转站过载


    max_token = MAX_TOKENS

    # ── 准备消息 ──────────────────────────────────────────
    messages = prepare_qa_inputs(model_name, queries, prompt=prompt)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    check_stop_signal()
    if STOP_FLAG:
        print("Stop signal detected before starting, exiting...")
        return

    # ── 回调：每完成一个请求就保存 ─────────────────────────
    async def save_on_complete(response, query_index):
        queries[query_index]["response"] = response
        queries[query_index]["model_name"] = model_name

        # 提取答案
        if response and isinstance(response, str) and response.strip():
            bracket_match = re.search(r'\[\s*(\w+)\s*\]', response.strip())
            if bracket_match:
                answer = bracket_match.group(1).strip()
            else:
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

        # 写文件
        async with SAVE_LOCK:
            check_stop_signal()
            if STOP_FLAG:
                print("Received stop signal, saving and exiting...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(queries, f, indent=4, ensure_ascii=False)
            if STOP_FLAG:
                print(f"Saved to {output_path}, exiting.")
                os._exit(0)

    # ── 异步执行并实时保存 ────────────────────────────────
    responses = await generate_from_claude_chat_completion_with_callback(
        client,
        messages=messages,
        engine_name=model_name,
        max_tokens=max_token,
        temperature=GENERATION_TEMPERATURE,
        requests_per_minute=base_rate,
        n=n,
        callback=save_on_complete,
        queries=queries,
    )

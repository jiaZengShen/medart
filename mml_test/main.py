from tqdm import tqdm
import json
import argparse
import os
import sys
import asyncio
from utils.constant import COT_PROMPT
from transformers.utils import logging
logging.set_verbosity_error() 

def main(
    model_name: str, 
    prompt: str, 
    queries: list, 
    output_path: str, 
    n: int=1, 
    api_base: str="")-> None:
    if (model_name.startswith("qwen")
            or model_name.startswith("doubao")
            or model_name.startswith("google/medgemma-1.5-4b-it")
            or model_name.startswith("glm-5.1")
            or model_name.startswith("kimi-k2.6")
            or model_name.startswith("minimax-m2.7")
            or model_name.startswith("Baichuan-M3-Plus")
            or model_name.startswith("medgemma-1.5-4b-it")
            or model_name.startswith("gpt-5.5")
            or model_name.startswith("gpt-5.4")
            or model_name.startswith("gpt-4o")
            or model_name.startswith("gemini-3.1")
    ):
        from model_inference.openai_api import generate_response
        asyncio.run(generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries,
                    output_path=output_path,
                    n = n,
                    api_base=api_base))
    elif "claude" in model_name:
        from model_inference.claude_api import generate_response
        asyncio.run(generate_response(model_name=model_name,
                                      prompt=prompt,
                                      queries=queries,
                                      output_path=output_path,
                                      n=n,
                                      api_base=api_base))
    # elif "gpt" in model_name:
    #     from model_inference.azure_gpt import generate_response
    #     generate_response(model_name=model_name,
    #                 prompt=prompt,
    #                 queries=queries,
    #                 output_path=output_path,
    #                 n = n,
    #                 api_base=api_base)
    elif "gemini" in model_name:
        from model_inference.openai_compatible import generate_response
        generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    output_path=output_path,
                    n = n,
                    api_base=api_base)
    elif "ollama" in model_name:
        from model_inference.ollama_inference import generate_response
        generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    output_path=output_path,
                    n = n,
                    api_base=api_base)
    elif model_name in json.load(open("model_inference/vllm_model_list.json")):
        from model_inference.vllm_inference import generate_response
        generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    output_path=output_path,
                    n = n,
                    api_base=api_base)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
        
prompt_dict = {
    "cot": COT_PROMPT,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="cot")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--max_num",type=int,default=0)
    parser.add_argument("--n",type=int,default=1)
    parser.add_argument("--times",type=int,default=1,
                        help="运行次数，>1 时每次输出保存为独立文件 xxx_1.json / xxx_2.json ...")
    parser.add_argument("--overwrite",action="store_true", default=False)
    parser.add_argument("--replace_image_prefix",type=str,default="",
                        help="替换 image_paths 的路径前缀，格式: old_prefix=new_prefix。例如: 'I:\\icdInformations\\1数据集=/new/path'")
    parser.add_argument("--model_path",type=str,default="",
                        help="覆盖模型权重根目录路径，默认使用 constant.py 中的配置")

    args = parser.parse_args()

    # 动态覆盖 MODEL_PATH（用于服务器等其他环境）
    if args.model_path:
        import utils.constant as _const
        _const.MODEL_PATH = args.model_path
    model_name = args.model

    try:
        prompt = prompt_dict[args.prompt]
    except KeyError:
        print("Invalid prompt")
        sys.exit(1)


    os.makedirs(args.output_dir, exist_ok=True)
        
    # 使用os.path模块处理路径，确保跨平台兼容
    data_path_basename = os.path.basename(args.data_path)
    data_path_suffix = os.path.splitext(data_path_basename)[0]
    output_dir = os.path.join(args.output_dir, f"{data_path_suffix}_{args.prompt}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_name = model_name.split("/")[-1]
    # 替换模型名称中的冒号，避免文件路径问题
    output_name = output_name.replace(":", "_")

    num_runs = args.times

    for run_idx in range(1, num_runs + 1):
        # 多次运行时每次保存为独立文件 xxx_1.json / xxx_2.json ...
        if num_runs > 1:
            output_path = os.path.join(output_dir, f"{output_name}_{run_idx}.json")
        else:
            output_path = os.path.join(output_dir, f"{output_name}.json")

        # 每次运行重新加载原始数据，避免复用上一次运行已写入的 response
        queries = json.load(open(args.data_path, "r", encoding="utf-8"))

        # 动态替换 image_paths 中的路径前缀
        if args.replace_image_prefix:
            if "=" not in args.replace_image_prefix:
                print("Error: --replace_image_prefix 格式错误，应为 old_prefix=new_prefix")
                sys.exit(1)
            old_prefix, new_prefix = args.replace_image_prefix.split("=", 1)
            for query in queries:
                if "image_paths" in query and isinstance(query["image_paths"], list):
                    query["image_paths"] = [
                        path.replace(old_prefix, new_prefix) if path.startswith(old_prefix) else path
                        for path in query["image_paths"]
                    ]

        if args.max_num > 0:
            queries = queries[:args.max_num]

        # 断点续传：非 overwrite 模式且输出文件已存在时，按 id 加载已有结果
        existing_results = {}
        if not args.overwrite and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for item in existing_data:
                        if 'id' in item:
                            existing_results[item['id']] = item
                print(f"已加载 {len(existing_results)} 条已有结果")
            except Exception as e:
                print(f"读取已有结果文件失败: {e}")

        # 非空 response 视作已回答
        answered_ids = {qid for qid, item in existing_results.items() if item.get('response')}

        # 所有 id 均已回答 → 该次运行已完成，跳过
        if answered_ids and all(q.get('id') in answered_ids for q in queries):
            print(f"跳过 {output_path}（已完成）")
            continue

        # 根据 id 合并数据（已回答的会被 api_utils 跳过，未回答的自动重新请求）
        if existing_results:
            for query in queries:
                query_id = query.get('id')
                if query_id in existing_results:
                    # 合并已有结果到 query 中
                    query.update(existing_results[query_id])
                    print(f"已合并 id: {query_id}")

        if os.path.exists(output_path) and args.overwrite:
            print(f"覆盖掉 {output_path}")

        print(f"=========Running {args.model} ({run_idx}/{num_runs})=========\n")
        try:
            main(
                model_name = args.model,
                prompt = prompt,
                queries = queries,
                output_path = output_path,
                n = args.n,
                api_base = args.api_base)
        except Exception as e:
            print(f"=========第 {run_idx}/{num_runs} 次运行失败: {e}=========")
            import traceback
            traceback.print_exc()
            if num_runs <= 1:
                raise
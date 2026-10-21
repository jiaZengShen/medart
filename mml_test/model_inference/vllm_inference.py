from utils.vllm_input_preparation import *
import json
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT

model_example_map = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":prepare_bioclip_inputs,#
    # "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2_inputs,
    # "Qwen/Qwen2-VL-72B-Instruct": prepare_qwen2_inputs,
    # "microsoft/Phi-3.5-vision-instruct": prepare_phi3v_inputs,
    # "microsoft/Phi-4-multimodal-instruct":prepare_phi3v_inputs,
    # "OpenGVLab/InternVL2-8B": prepare_general_vlm_inputs,
    # "OpenGVLab/InternVL2_5-8B": prepare_general_vlm_inputs,
    # "OpenGVLab/InternVL2_5-38B":prepare_general_vlm_inputs,
    # "mistral-community/pixtral-12b": prepare_pixtral_inputs,
    # "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision_inputs,
    # "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama_inputs,
    # "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2_inputs,
    # "OpenGVLab/InternVL3-8B":prepare_general_vlm_inputs,
    # "OpenGVLab/InternVL3-38B":prepare_general_vlm_inputs,
    # "Qwen/Qwen2.5-VL-72B-Instruct": prepare_qwen2_inputs,
    # "mistralai/Mistral-Small-3.1-24B-Instruct-2503": prepare_pixtral_inputs,
}

vllm_model_list = list(model_example_map.keys())
json.dump(vllm_model_list, open("model_inference/vllm_model_list.json", "w"), indent=4, ensure_ascii=False)

def generate_vlm_response(model_name,
                       queries,
                       prompt=COT_PROMPT,
                       temperature: float=1,
                       max_tokens: int=1024):
    result = model_example_map[model_name](model_name, queries, prompt, temperature, max_tokens)

    if len(result) == 3:
        # vllm 模式: (inputs, llm, sampling_params)
        inputs, llm, sampling_params = result
        responses = llm.generate(inputs, sampling_params=sampling_params)
        return [response.outputs[0].text for response in tqdm(responses, desc="Inference")]
    elif len(result) == 4:
        # transformers 模式: (model, processor, prepared_inputs, generation_kwargs)
        import torch
        model, processor, prepared_inputs, generation_kwargs = result

        # 对比式模型（如 BiomedCLIP）：prepared_inputs 已经是预计算的文本回答列表
        if prepared_inputs and isinstance(prepared_inputs[0], str):
            return prepared_inputs

        device = next(model.parameters()).device
        responses = []
        for item_input in tqdm(prepared_inputs, desc="推理"):
            item_input = {k: v.to(device) if hasattr(v, 'to') else v for k, v in item_input.items()}
            with torch.no_grad():
                outputs = model.generate(**item_input, **generation_kwargs)
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)
            responses.append(decoded[0])
            # 每个样本推理后释放GPU缓存，防止显存碎片累积导致OOM
            if torch.cuda.is_available():
                del outputs
                torch.cuda.empty_cache()
        return responses
    else:
        raise ValueError(f"Unsupported return format from prepare function for {model_name}")

def generate_response(model_name: str,
                    prompt: str,
                    queries: list,
                    output_path: str,
                    n: int=1,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS,
                    api_base: str=""):
    if model_name not in model_example_map:
        raise ValueError(f"Model type {model_name} is not supported.")
    responses = generate_vlm_response(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      temperature=temperature, 
                                      max_tokens=max_tokens)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT

from argparse import Namespace
from typing import List
import torch

import os
from tqdm import tqdm
from utils.input_processing import prepare_qa_text_input, prepare_single_image_input,prepare_image_input
from utils.constant import *

def prepare_bioclip_inputs(
        model_name,
        queries,
        prompt,
        temperature: float = 1,
        max_tokens: int = 1024
):
    """
    Prepare inputs + run inference for BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.

    BiomedCLIP is a *contrastive* vision-language model (CLIP-like). It does NOT
    generate text autoregressively. Instead, it encodes images and text into a
    shared embedding space and computes cosine similarity.

    For VQA tasks we:
      1. Encode the image(s) via the vision encoder.
      2. Encode candidate answers ("Yes", "No") via the text encoder.
      3. Pick the candidate with the highest similarity.

    Returns
    -------
    (model, processor, responses, {})
        responses : list[str]  — chosen candidate ("Yes" / "No") per query.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    model_path = MODEL_PATH + model_name

    # ── 自动检测 GPU 可用显存 ──────────────────────────────────
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        gpu_reserved = 1.0
        gpu_available_gib = max(gpu_total / (1024**3) - gpu_reserved, 0.5)
        gpu_for_model = f"{gpu_available_gib * 0.5:.1f}GiB"
        max_mem = {0: gpu_for_model, "cpu": "48GiB"}
        print(f"[BiomedCLIP] GPU分配给模型: {gpu_for_model}, 其余层offload到CPU")
    else:
        max_mem = {"cpu": "48GiB"}
        print("[BiomedCLIP] 纯CPU推理")

    # ── 加载模型和 processor（BiomedCLIP 需要 trust_remote_code） ──
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_mem,
        offload_folder="offload",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    device = next(model.parameters()).device

    # ── 对每个 query 计算图文相似度 ────────────────────────────
    responses = []
    for query in tqdm(queries, desc="BiomedCLIP inference"):
        _, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        candidate_texts = [
            f"{qa_text_prompt} Answer: Yes",
            f"{qa_text_prompt} Answer: No",
        ]

        # 加载图片
        image_paths = query.get("image_paths", [])
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        # 图片编码 → pixel_values
        image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
        pixel_values = image_inputs["pixel_values"]

        # 候选文本编码 → input_ids, attention_mask
        text_inputs = processor(
            text=candidate_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # 前向传播计算相似度
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            logits = outputs.logits_per_image  # [num_images, 2]

            # 多图场景：平均各图的 logits，然后取 argmax
            logits = logits.mean(dim=0, keepdim=True)
            pred_idx = logits.argmax(dim=-1).item()

        responses.append(candidate_texts[pred_idx])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, processor, responses, {}


# def prepare_qwen2_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA inputs for Qwen2-VL-7B-Instruct or similar models.
#     """
#     inputs = []
#     llm = LLM(
#         model=model_name,
#         tensor_parallel_size=min(torch.cuda.device_count(), 4),
#         # max_model_len=49152,
#         limit_mm_per_prompt={"image": 2}  # Only allow 1 image
#     )
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=None
#     )
#
#     processor = AutoProcessor.from_pretrained(
#         model_name,
#         min_pixels=256*28*28,
#         max_pixels=1280*28*28
#     )
#
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         # 1. Prepare text QA prompt
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # 2. Retrieve single base64 image (using your custom function)
#         vision_input = prepare_image_input(model_name, query)
#         urls = []
#         if query['claim_type']=='direct' or query['claim_type']=='analytical':
#             urls.append(query['image_path'])
#         else:
#             urls.append(query['item1_path'])
#             urls.append(query['item2_path'])
#
#         placeholders = [{"type": "image", "image": url} for url in urls]
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     *placeholders,
#                     {"type": "text", "text": qa_text_prompt},
#                 ],
#             }
#         ]
#
#         # Use Qwen's AutoProcessor to form a proper text input with generation prompt
#         text_input = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#
#         item_input = {
#             "prompt": text_input,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     return inputs, llm, sampling_params
#
#
# def prepare_phi3v_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA inputs for Microsoft Phi-3.5-vision-instruct or similar models.
#     """
#     inputs = []
#     llm = LLM(
#         model=model_name,
#         trust_remote_code=True,
#         limit_mm_per_prompt={"image": 2},
#         tensor_parallel_size=min(torch.cuda.device_count(),4),
#     )
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=None
#     )
#
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         # QA text
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # Convert base64 to actual image object
#         # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
#         # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
#         vision_input = prepare_image_input(model_name, query)
#
#         # Placeholders for Phi-3.5 format
#         placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(vision_input, start=1))
#         text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"
#
#         item_input = {
#             "prompt": text_input,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     return inputs, llm, sampling_params
#
#
# def prepare_general_vlm_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA for a general VLM-based model (e.g., some HF large models).
#     """
#     inputs = []
#
#     if "molmo" in model_name.lower():
#         max_model_len = 4096
#         llm = LLM(
#             model=model_name,
#             trust_remote_code=True,
#             dtype="bfloat16",
#             max_model_len=max_model_len,
#             limit_mm_per_prompt={"image": 2},
#             tensor_parallel_size=min(torch.cuda.device_count(),4),
#         )
#     else:
#         # max_model_len = 65536
#         max_model_len = 1024*32
#         llm = LLM(
#             model=model_name,
#             trust_remote_code=True,
#             max_model_len=max_model_len,
#             limit_mm_per_prompt={"image": 2},
#             tensor_parallel_size=min(torch.cuda.device_count(),4),
#         )
#
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         # QA text
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # Single image
#         # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
#         # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
#         vision_input = prepare_image_input(model_name, query)
#         placeholders = "\n".join(f"Image-{i}: <image>\n"
#                                  for i, _ in enumerate(vision_input, start=1))
#         messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
#
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         text_input = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#
#         item_input = {
#             "prompt": text_input,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     if "h2oai" in model_name:
#         stop_token_ids = [tokenizer.eos_token_id]
#     else:
#         stop_token_ids = None
#
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=stop_token_ids
#     )
#
#     return inputs, llm, sampling_params
#
#
# def prepare_pixtral_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA for Pixtral models.
#     """
#     stop_token_ids = None
#     llm = LLM(
#         model=model_name,
#         max_model_len=8192*4*4,
#         max_num_seqs=2,
#         limit_mm_per_prompt={"image": 2},
#         tensor_parallel_size=min(torch.cuda.device_count(),4)
#     )
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=stop_token_ids
#     )
#
#     inputs = []
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         # QA text
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
#         vision_input = prepare_image_input(model_name, query)
#         placeholders = "[IMG]" * len(vision_input)
#         prompt_info = f"<s>[INST]{qa_text_message}\n{placeholders}[/INST]"
#         # For Pixtral, the original logic places "[IMG]" placeholders
#         # cur_input = f"<s>[INST]{qa_text_prompt}\n[IMG][/INST]"
#         item_input = {
#             "prompt": prompt_info,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     return inputs, llm, sampling_params
#
#
# def prepare_mllama_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA for M-LLaMA or related models.
#     """
#     inputs = []
#     llm = LLM(
#         model=model_name,
#         limit_mm_per_prompt={"image": 2},
#         max_model_len=8192*4,
#         max_num_seqs=2,
#         enforce_eager=True,
#         trust_remote_code=True,
#         tensor_parallel_size=min(torch.cuda.device_count(),4),
#     )
#
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=[128001,128008,128009]
#     )
#
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         # QA text
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
#         # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
#         vision_input = prepare_image_input(model_name, query)
#
#         # placeholders = "<|image|>"*len(vision_input)
#         placeholders = " ".join(["<|image|>"] * len(vision_input))
#         messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
#
#         # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         # input_prompt = tokenizer.apply_chat_template(
#         #     messages,
#         #     tokenize=False,
#         #     add_generation_prompt=True
#         # )
#         input_prompt = f"<|begin_of_text|>{placeholders}\n{qa_text_prompt}"
#         item_input = {
#             "prompt": input_prompt,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     return inputs, llm, sampling_params
#
#
# def prepare_llava_onevision_inputs(
#     model_name,
#     queries,
#     prompt,
#     temperature: float = 1,
#     max_tokens: int = 1024
# ):
#     """
#     Prepare single-image QA for LLaVA-OneVision.
#     """
#     inputs = []
#     for query in tqdm(queries, desc="Prepare model inputs"):
#         qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#
#         # Only <image> placeholder
#         vision_input = prepare_image_input(model_name, query)
#         placeholders = " ".join(["<image>"] * len(vision_input))
#         input_prompt = f"<|im_start|>user {placeholders}\n{qa_text_prompt}<|im_end|>\n<|im_start|>assistant\n"
#
#         # single_img_b64 = prepare_single_image_input(model_name, query['image_path'])
#         # vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]
#
#
#         item_input = {
#             "prompt": input_prompt,
#             "multi_modal_data": {
#                 "image": vision_input
#             },
#         }
#         inputs.append(item_input)
#
#     llm = LLM(
#         model=model_name,
#         max_model_len=32768,
#         limit_mm_per_prompt={"image": 2},
#         tensor_parallel_size=min(torch.cuda.device_count(),4)
#     )
#
#     stop_token_ids = None
#     sampling_params = SamplingParams(
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stop_token_ids=stop_token_ids
#     )
#
#     return inputs, llm, sampling_params

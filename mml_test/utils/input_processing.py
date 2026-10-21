import base64
from utils.constant import COT_PROMPT
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64
import json
from PIL import Image
from io import BytesIO

# 替代vllm的fetch_image函数
def fetch_image(image_url):
    """
    Fetch an image from a URL or base64 string.
    Args:
        image_url: A URL or base64 string representing the image.
    Returns:
        The image content as bytes.
    """
    if image_url.startswith('data:image'):
        # 如果是base64字符串，提取base64数据
        base64_data = image_url.split(',')[1]
        return base64.b64decode(base64_data)
    else:
        # 如果是URL，下载图像
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content

def prepare_context(paper_path,section_list):
    with open(paper_path,'r',encoding='utf-8')as f:
        paper = json.load(f)
    top_sections = []
    for item in section_list:
        if item.split('.')[0] not in top_sections:
            top_sections.append(item.split('.')[0])
    context = ""
    for sec in paper['sections']:
        if sec['section_id'].split('.')[0] in top_sections:
            context += sec['section_name'] + ':\n' + sec['text'] + '\n'
    return context


#准备文本
def prepare_qa_text_input(model_name, query, prompt):
    claim_type = query.get('claim_type', 'direct')
    # 使用claim_deepseek作为声明，如果不存在则使用claim
    claim = query.get('claim')

    user_prompt = prompt.get(claim_type, prompt.get('recognition', prompt))
    # 直接从JSON文件中读取context字段
    context = query.get('context')
    # 提供默认值字典，避免KeyError
    default_values = {
        'claim': claim,
        'context': context
    }
    qa_text_prompt = user_prompt.substitute(default_values)
    qa_text_message = {
        "type": "text",
        "text": qa_text_prompt
    }
    return qa_text_message, qa_text_prompt

def prepare_single_image_input(model_name, image_source):
    """
    Prepare a single image input for different model requirements.

    This function now accepts either:
      - A local image file path (e.g., "path/to/image.jpg")
      - An image URL (e.g., "http://.../some_image.jpg" or "https://...")
    
    It then converts the image content into base64 internally before returning
    the final structure, which varies depending on 'model_name':
      - a dict with {"type": "image", "source": {...}} for Claude
      - a raw base64 string for vLLM or other HF-based models
      - a dict with {"type": "image_url", ...} otherwise

    Args:
        model_name (str): The name of the model. Used to decide the returned format.
        image_source (str): Either a local file path or an image URL.

    Returns:
        A dictionary or a base64 string, depending on the model_name.
    """

    # 1. Determine if the source is a local path or a URL
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # If it's an HTTP/HTTPS URL, download the image
        response = requests.get(image_source)
        response.raise_for_status()
        image_content = response.content
    else:
        # Otherwise, assume it's a local file path
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        with open(image_source, "rb") as f:
            image_content = f.read()
    # if model_name!='Qwen/Qwen2-VL-7B-Instruct' and model_name!='microsoft/Phi-4-multimodal-instruct':
    #     min_size = 224
    #     # 2. 用 PIL 打开图像并检查大小
    #     image = Image.open(BytesIO(image_content)).convert("RGB")
    #     if image.width < min_size or image.height < min_size:
    #         # 比例保持不变地放大到最小边 min_size
    #         ratio = max(min_size / image.width, min_size / image.height)
    #         new_size = (int(image.width * ratio), int(image.height * ratio))
    #         image = image.resize(new_size, Image.BICUBIC)
    # else:
    #     min_size = 224
    #     image = Image.open(BytesIO(image_content)).convert("RGB")
    #     width, height = image.size
    #     # 缩放比例，使得最小边变为 224
    #     scale = min_size / min(width, height)
    #
    #     # 计算新尺寸
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    #
    #     # Resize
    #     image = image.resize((new_width, new_height), Image.BICUBIC)
    image = Image.open(BytesIO(image_content)).convert("RGB")

    # 3. 转回字节内容
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_content = buffer.getvalue()

    # 2. Convert the image content to base64
    image_base64 = base64.b64encode(image_content).decode("utf-8")
    # 3. Return the structure based on the model_name
    if "claude" in model_name:
        # Claude usually requires {"type":"image", "source":{...}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_base64,
            },
        }
    elif "ollama" in model_name:
        # For Ollama models, return image_url format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}",
            },
        }
    # elif "/" in model_name:
    #     # For vLLM or other HF-based models, return raw base64
    #     return image_base64
    else:
        # Some fallback or custom format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}",
            },
        }

def prepare_image_input(model_name, query):
    # 处理image_paths数组
    image_paths = query.get('image_paths', [])
    vision_input = []
    for image_path in image_paths:
        single_img_b64 = prepare_single_image_input(model_name, image_path)
        # 由于prepare_single_image_input函数已经将所有图片转换为JPEG格式，所以使用data:image/jpeg是安全的
        vision_input.append(fetch_image(f"data:image/jpeg;base64,{single_img_b64}"))
    return vision_input

# 准备问答
def prepare_qa_inputs(model_name, queries, prompt=COT_PROMPT):
    """
    Prepare the final list of messages that combine single image input with QA text.
    - Each query is turned into a list containing a user role with two items: image + text QA.
    """
    messages = []
    for query in tqdm(queries):
        # Create the QA text portion
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Create or retrieve the single image data structure
        # 处理image_paths数组
        image_paths = query.get('image_paths', [])
        image_data_list = []
        for image_path in image_paths:
            # 检查image_path是否为空
            if image_path and isinstance(image_path, str) and image_path.strip():
                try:
                    image_data = prepare_single_image_input(model_name, image_path)
                    image_data_list.append(image_data)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    # 跳过有问题的图像，继续处理下一个
                    continue

        # Combine them into a "user" content list
        # 如果image_data_list为空，则只包含文本消息
        if image_data_list:
            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        *image_data_list,
                        qa_text_message
                    ],
                }
            ]
        else:
            # 如果没有图像，则只包含文本消息
            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        qa_text_message
                    ],
                }
            ]
        messages.append(prompt_message)
    return messages

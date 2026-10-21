"""
Image flip tool for LangChain agent — flips an image horizontally (left↔right).

提供图像左右翻转能力。在本项目的医学影像 claim 验证场景中，侧别
（left / right）是关键判断点：当怀疑侧别标注或观察有误时，可将图像左右
翻转后让 LLM 重新"看"一遍，辅助交叉验证。

遵循 biomedparse_client.py / web_fetch_tool.py 的模式：
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组
  - content_blocks 包含翻转后图像的 data URL，让多模态 LLM 直接看到
  - result 为应用层元数据（路径、尺寸、状态）
- 结果持久化到本地目录

结果持久化（本地保存，辅助调试 / 审计）：
- flipped/{timestamp}_{name}_flipped{ext}   翻转后的图像文件
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from langchain.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT
from src.multi_agent.image_utils import build_image_data_url, flip_image_file


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class ImageFlipInput(BaseModel):
    image_path: str = Field(
        description="Input: Absolute path to the local image to flip horizontally.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=ImageFlipInput)
def flip_image_horizontally(
    image_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flip an image horizontally (mirror left ↔ right) and return the result.

    Use this tool when you need to re-examine an image with its left and right
    sides swapped — most commonly to double-check the laterality (left/right
    side) of a finding in a medical image. The flipped image is returned so you
    can directly view it.

    Inputs:
    - image_path: Absolute path to the local image file to flip

    Outputs (content_blocks → visible to LLM):
    - Text: Flip summary (input path, output path, dimensions)
    - Image: The horizontally flipped image as a data URL

    Outputs (result dict → application metadata):
    - success: Whether the flip succeeded
    - input_path: The original image path
    - output_path: Path where the flipped image was saved
    - width / height: Image dimensions in pixels
    - message: Status description or error message
    """

    def _error_return(msg: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"Image flip failed: {msg}"}
        ]
        result: dict[str, Any] = {
            "success": False,
            "input_path": image_path,
            "output_path": "",
            "width": 0,
            "height": 0,
            "message": msg,
        }
        return content_blocks, result

    # =====================================================================
    # 1. 校验输入路径
    # =====================================================================
    src = Path(image_path)
    if not image_path or not src.exists():
        return _error_return(f"Image file not found: '{image_path}'")
    if not src.is_file():
        return _error_return(f"Path is not a file: '{image_path}'")

    # =====================================================================
    # 2. 读取尺寸并水平翻转（翻转与保存交由共享纯函数处理）
    # =====================================================================
    try:
        with Image.open(src) as img:
            width, height = img.size
    except Exception as e:
        traceback.print_exc()
        return _error_return(f"Failed to open image '{image_path}': {e}")

    try:
        out_dir = _ensure_dir(PROJECT_ROOT / "flipped")
        save_path = flip_image_file(image_path, out_dir=out_dir)
    except Exception as e:
        traceback.print_exc()
        return _error_return(f"Failed to flip/save image '{image_path}': {e}")

    # =====================================================================
    # 4. 构建 data URL 供 LLM 查看
    # =====================================================================
    try:
        data_url = build_image_data_url(str(save_path))
    except Exception as e:
        traceback.print_exc()
        return _error_return(
            f"Image flipped and saved but failed to encode data URL: {e}"
        )

    # =====================================================================
    # 5. 构建 LLM content blocks（文本 + 翻转图像）
    # =====================================================================
    content_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"## Image Flipped Horizontally (left ↔ right)\n\n"
                f"**Input**: {image_path}\n"
                f"**Output**: {save_path}\n"
                f"**Dimensions**: {width}×{height}px\n\n"
                f"The following is the flipped image — left and right are "
                f"mirrored relative to the original:\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]

    # =====================================================================
    # 6. 构建 result dict
    # =====================================================================
    result_dict: dict[str, Any] = {
        "success": True,
        "input_path": str(src),
        "output_path": str(save_path),
        "width": width,
        "height": height,
        "message": (
            f"Flipped '{image_path}' horizontally ({width}×{height}px), "
            f"saved to {save_path}"
        ),
    }

    return content_blocks, result_dict

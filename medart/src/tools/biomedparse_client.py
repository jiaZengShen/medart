"""
BiomedParse V2 segmentation tool for LangChain agent.

Sends 2D pathology / radiology images to the BiomedParse segmentation server
and returns:
  1. Segmentation mask (binary PNG)
  2. Mask overlay on original image (red semi-transparent)
both as data URLs for the LLM.

遵循 medsam_tool.py 的模式：
- 服务端返回 .npy → 转 PNG 保存 → data URL 给 LLM
- 返回 (content_blocks, result) 元组
- 所有路径返回格式一致

IoU / Dice 等分割指标不在工具内计算（LLM 无法提供真实掩码路径），
通过 calculate_iou(), calculate_dice(), load_gt_mask() 外部调用。"""

from __future__ import annotations

import base64
import io
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from langchain.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from src.config import BIOMEDPARSE_SERVER_URL, DEFAULT_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Helper: numpy mask → PNG data URL
# ---------------------------------------------------------------------------


def _mask_to_png_data_url(pred_mask: np.ndarray) -> str:
    """Convert binary probability mask (H, W) to PNG data URL."""
    # 二值化（>0.5 → 白，≤0.5 → 黑）
    binary = (np.clip(pred_mask, 0, 1) > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(binary, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _build_data_url(buf.getvalue())


def _build_data_url(png_bytes: bytes) -> str:
    """Convert PNG binary content to data URL for LLM input."""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection-over-Union between a probability map and a binary ground truth.

    Parameters
    ----------
    pred_mask : np.ndarray
        Float32 probability map (values in [0, 1]).
    gt_mask : np.ndarray
        Binary ground-truth mask (0 / 1, dtype uint8 or bool).

    Returns
    -------
    float
        IoU score, 0.0 if union is empty.
    """
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    intersection = (pred_binary & gt_mask).sum()
    union = (pred_binary | gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between a probability map and a binary ground truth."""
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    intersection = (pred_binary & gt_mask).sum()
    denom = pred_binary.sum() + gt_mask.sum()
    return (2.0 * intersection) / denom if denom > 0 else 0.0


def load_gt_mask(gt_mask_path: str) -> np.ndarray | None:
    """
    Load a binary ground-truth mask from a PNG image file.

    Parameters
    ----------
    gt_mask_path : str
        Absolute path to the ground-truth mask image.

    Returns
    -------
    np.ndarray or None
        uint8 binary mask (0 / 1) or None if the file is missing/corrupt.
    """
    path = Path(gt_mask_path)
    if not path.exists():
        return None
    try:
        gt_img = Image.open(path).convert("RGB")
        return (1 * (np.array(gt_img)[:, :, 0] > 0)).astype(np.uint8)
    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class BiomedParseSegmentInput(BaseModel):
    image_path: str = Field(
        description="Input: Absolute path to the pathology/radiology image to segment, required."
    )
    prompt: str = Field(
        description=(
            "Required: The exact anatomical structure name to segment. "
        "MUST be a single anatomical structure (e.g., 'left basal ganglia', 'pons', 'infraspinatous muscle'). "
        "DO NOT use pathological descriptions (e.g., 'tumor', 'lesion', 'abnormal signal'). "
        "If the target is not an anatomical structure, do not call this tool."
        ),
    )
    case_id: str | None = Field(
        default=None,
        description="Input: Sample or case identifier, optional, used for naming output mask files.",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=BiomedParseSegmentInput)
def biomedparse_segment(
    image_path: str,
    prompt: str,
    case_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Segment pathology / radiology images using the BiomedParse V2

    **MANDATORY USE CASES — YOU HAVE NO CHOICE**:
    - The claim references a SINGLE, CLEAR anatomical structure (e.g., "pons", "left basal ganglia", "infraspinatous muscle")
    - You need to confirm whether this structure EXISTS in the image
    - You need to determine the structure's SIDE (left/right)
    — YOU MUST CALL THIS TOOL

    **RECOMMENDED USE CASES**:
    - You want objective spatial evidence to support your judgment
    - You need to confirm what you observed visually
    - The claim requires precise localization of any target structure
    - You want to be thorough and rigorous

    DO NOT USE THIS TOOL IF:
    - Truly trivial cases with NO spatial aspects, NO side, NO negation, AND the finding is overwhelmingly obvious

    Inputs:
    - image_path: Absolute path to the pathology/radiology image to segment, required
    - prompt: Text description of the target to segment, e.g., 'neoplastic cells', required
    - case_id: Sample or case identifier, optional, used for naming output files

    Outputs (content_blocks → visible to LLM):
    - Text description
    - Overlay image: mask overlaid on original image (red semi-transparent, data URL)
    - Mask image: segmentation mask only (data URL)

    Outputs (result dict → application metadata):
    - mask_path: Path where the segmentation mask PNG file is saved
    - overlay_path: Path where the mask+original overlay PNG file is saved
    - success: Whether the operation succeeded
    - message: Status description
    """
    # 1. 确定服务地址
    base_url = BIOMEDPARSE_SERVER_URL
    segment_url = f"{base_url.rstrip('/')}/segment"
    health_url = f"{base_url.rstrip('/')}/health"

    # 2. Validate input file
    if not Path(image_path).exists():
        msg = f"Image file not found: {image_path}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    # 3. 健康检查（快速失败）
    try:
        resp = requests.get(health_url, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        msg = f"Cannot connect to BiomedParse service: {base_url}, please verify the service is running"
        return _error_content_blocks(msg), {"success": False, "message": msg}
    except requests.exceptions.RequestException as e:
        msg = f"BiomedParse service health check failed: {e}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    # 4. 调用分割接口
    try:
        resp = requests.post(
            segment_url,
            json={"image_path": image_path, "captions": prompt},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        msg = f"BiomedParse segmentation request failed: {e}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    # 5. Parse the returned .npy mask
    try:
        pred_mask: np.ndarray = np.load(io.BytesIO(resp.content))
    except Exception as e:
        msg = f"Failed to parse .npy mask data from service: {e}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    # 6. 确定输出路径
    output_path_base = Path(str(DEFAULT_OUTPUT_DIR)) / "biomedparse_masks"
    output_path_base.mkdir(parents=True, exist_ok=True)

    # 获取原始图像文件名（不含扩展名）
    original_image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if case_id:
        base_filename = f"{original_image_name}_{case_id}_{timestamp}"
    else:
        base_filename = f"{original_image_name}_{timestamp}"

    filename = f"{base_filename}_mask.png"
    overlay_filename = f"{base_filename}_overlay.png"

    mask_path = output_path_base / filename
    overlay_path = output_path_base / overlay_filename

    # 7. 将 numpy mask 转成 PNG 字节 → 保存 + data URL
    png_bytes = _mask_to_png_bytes(pred_mask)
    try:
        mask_path.write_bytes(png_bytes)
    except Exception as e:
        msg = f"Failed to save mask file: {e}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    mask_data_url = _build_data_url(png_bytes)

    # 7b. 生成并保存融合图像（mask 叠加在原图上）
    overlay_bytes = _overlay_mask_on_image(image_path, pred_mask, alpha=0.5, color=(255, 0, 0))
    try:
        overlay_path.write_bytes(overlay_bytes)
    except Exception as e:
        msg = f"Failed to save overlay image: {e}"
        return _error_content_blocks(msg), {"success": False, "message": msg}

    overlay_data_url = _build_data_url(overlay_bytes)

    # 8. Build result dict
    result: dict[str, Any] = {
        "success": True,
        "message": f"Segmentation completed, mask saved to {mask_path}, overlay saved to {overlay_path}",
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }

    # 9. Content blocks for the LLM: text + mask image + overlay image
    content_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"BiomedParse segmentation completed:\n"
                f"- Status: Success\n"
                f"- Mask saved at: {mask_path}\n"
                f"- Overlay saved at: {overlay_path}\n"
                f"- Target description: {prompt}\n"
                f"- Here are the segmentation results (mask only + overlay on original image):"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": overlay_data_url},
        },
        {
            "type": "text",
            "text": "Segmentation mask only:",
        },
        {
            "type": "image_url",
            "image_url": {"url": mask_data_url},
        },
    ]

    return content_blocks, result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mask_to_png_bytes(pred_mask: np.ndarray) -> bytes:
    """Convert a binary probability mask (H, W) to PNG bytes."""
    binary = (np.clip(pred_mask, 0, 1) > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _overlay_mask_on_image(
    image_path: str,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> bytes:
    """
    Overlay segmentation mask on original image with transparency.

    Parameters
    ----------
    image_path : str
        Path to original input image
    pred_mask : np.ndarray
        Binary probability mask (H, W)
    alpha : float
        Transparency of mask overlay (0.0 - 1.0)
    color : tuple
        RGB color for mask overlay (default red)

    Returns
    -------
    bytes
        PNG bytes of overlay image
    """
    # Load original image
    orig_img = Image.open(image_path).convert("RGB")
    orig_arr = np.array(orig_img)

    # Resize mask if needed to match image size
    mask_h, mask_w = pred_mask.shape[:2]
    img_h, img_w = orig_arr.shape[:2]

    if (mask_h, mask_w) != (img_h, img_w):
        mask_img = Image.fromarray((np.clip(pred_mask, 0, 1) > 0.5).astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
        binary = np.array(mask_img)
    else:
        binary = (np.clip(pred_mask, 0, 1) > 0.5).astype(np.uint8) * 255

    # Create color mask
    color_mask = np.zeros_like(orig_arr)
    color_mask[binary > 128] = color

    # Blend images
    overlay_arr = orig_arr.copy()
    mask_region = binary > 128
    overlay_arr[mask_region] = (
        orig_arr[mask_region] * (1 - alpha) + color_mask[mask_region] * alpha
    ).astype(np.uint8)

    overlay_img = Image.fromarray(overlay_arr)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return buf.getvalue()


def _error_content_blocks(message: str) -> list[dict[str, Any]]:
    """Build a text-only content block for error returns."""
    return [{"type": "text", "text": f"BiomedParse segmentation failed: {message}"}]


# ---------------------------------------------------------------------------
# Side Analysis Integration
# ---------------------------------------------------------------------------


def biomedparse_segment_with_side_analysis(
    image_path: str,
    prompt: str,
    case_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Segment an image using BiomedParse AND perform side distribution analysis.

    This is a convenience wrapper that:
    1. Calls biomedparse_segment() to get the mask
    2. Uses side_analyzer to analyze mask distribution
    3. Returns enriched result with side analysis included

    Args:
        image_path: Absolute path to input image
        prompt: Text description of target anatomy
        case_id: Optional identifier for output files

    Returns:
        (content_blocks, result) tuple where result includes side_analysis
    """
    # Import here to avoid circular dependencies
    from src.tools.side_analyzer import (
        load_mask,
        analyze_side_distribution,
        format_side_analysis_for_llm,
    )

    # First perform segmentation
    content_blocks, result = biomedparse_segment.func(
        image_path=image_path,
        prompt=prompt,
        case_id=case_id,
    )

    # If segmentation failed, return early
    if not result.get("success", False):
        result["side_analysis"] = None
        return content_blocks, result

    # Load the mask and analyze
    mask_path = result.get("mask_path")
    if mask_path:
        mask = load_mask(mask_path)
        side_analysis = analyze_side_distribution(mask)
        result["side_analysis"] = side_analysis

        # Also add formatted analysis text to content_blocks for LLM
        formatted_analysis = format_side_analysis_for_llm(side_analysis)

        # Insert after the first text block
        if content_blocks and len(content_blocks) > 0:
            # Find the first text block to insert after
            inserted = False
            for i, block in enumerate(content_blocks):
                if block.get("type") == "text":
                    # Insert analysis after this text block
                    content_blocks.insert(i + 1, {
                        "type": "text",
                        "text": f"\n---\n\n{formatted_analysis}"
                    })
                    inserted = True
                    break
            if not inserted:
                content_blocks.append({
                    "type": "text",
                    "text": f"\n---\n\n{formatted_analysis}"
                })
    else:
        result["side_analysis"] = None

    return content_blocks, result


# ---------------------------------------------------------------------------
# Standalone entry point（向后兼容，方便直接调试）
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("===== BiomedParse V2 Client (standalone) =====")

    import sys

    test_image = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I:\\icdInformations\\1数据集\\HC18/training_set/000_HC.png"
    )
    test_prompt = sys.argv[2] if len(sys.argv) > 2 else "fetal head"

    print(f"Image:  {test_image}")
    print(f"Prompt: {test_prompt}")
    print("-" * 50)

    content_blocks, result = biomedparse_segment.func(
        image_path=test_image,
        prompt=test_prompt,
    )

    print("\n[content_blocks]")
    for block in content_blocks:
        if block["type"] == "text":
            print(block["text"][:200])
        elif block["type"] == "image_url":
            print(f"[image data URL: {block['image_url']['url'][:60]}...]")

    print("\n[result dict]")
    for k, v in result.items():
        print(f"  {k}: {v}")

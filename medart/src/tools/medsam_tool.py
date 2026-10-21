from typing import Any

import base64
import mimetypes
import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.config import MEDSAM_SERVER_URL
from datetime import datetime
from pathlib import Path

class MedSAMSegmentInput(BaseModel):
    image_path: str = Field(description="Input: Absolute path to the image to be segmented, required.")
    bbox: list[float] = Field(
        description=(
            "Input: Target bounding box in XYXY pixel coordinates, required. "
            "Format: [x1, y1, x2, y2], e.g., [100, 80, 300, 320]."
        ),
    )
    case_id: str | None = Field(
        default=None,
        description="Input: Sample or case identifier, optional, used for naming output files.",
    )


def _build_image_data_url_from_bytes(png_bytes: bytes) -> str:
    """Convert PNG binary content to data URL for LLM input."""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@tool(args_schema=MedSAMSegmentInput)
def medsam_segment(
    image_path: str,
    bbox: list[float],
    case_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Segment a region of interest from an ultrasound image using the MedSAM HTTP server.

    Calls the remote MedSAM service via HTTP to segment the specified bbox region in the image,
    returning a binary mask to the LLM (including the mask image).

    Inputs:
    - image_path: Absolute path to the ultrasound image to segment, required
    - bbox: Target bounding box in XYXY pixel coordinates [x1, y1, x2, y2], required
    - case_id: Sample or case identifier, optional, used for naming output files

    Outputs:
    - mask_path: Path where the segmentation mask PNG file is saved
    - success: Whether the operation succeeded
    - message: Status description
    - mask_data_url: Data URL of the mask image for LLM viewing
    """
    # 1. 确定服务地址
    base_url =  MEDSAM_SERVER_URL
    segment_url = f"{base_url.rstrip('/')}/segment"
    health_url = f"{base_url.rstrip('/')}/health"
    output_dir = "./outputs/medsam/"

    # 2. 校验 bbox
    if len(bbox) != 4:
        return {
            "success": False,
            "message": f"bbox must be a 4-element list [x1, y1, x2, y2], received {len(bbox)} elements",
            "mask_path": None,
        }

    # 3. 健康检查（快速失败，避免上传大图后才发现服务不可用）
    try:
        resp = requests.get(health_url, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": f"Cannot connect to MedSAM service: {base_url}, please verify the service is running",
            "mask_path": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"MedSAM service health check failed: {e}",
            "mask_path": None,
        }

    # 4. 调用分割接口
    try:
        resp = requests.post(
            segment_url,
            json={"image_path": image_path, "bbox": bbox},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"MedSAM segmentation request failed: {e}",
            "mask_path": None,
        }

    # 5. 检查响应 Content-Type
    content_type = resp.headers.get("content-type", "")
    if "image/png" not in content_type:
        return {
            "success": False,
            "message": f"Service returned unexpected content type: {content_type}, there may be a middleware configuration issue",
            "mask_path": None,
        }

    # 6. 确定输出路径
    if output_dir is None:


        from src.config import DEFAULT_OUTPUT_DIR

        output_dir = str(DEFAULT_OUTPUT_DIR)

    output_path_base = Path(output_dir) / "medsam_masks"
    output_path_base.mkdir(parents=True, exist_ok=True)

    # 文件名：用 case_id 或时间戳
    if case_id:
        filename = f"mask_{case_id}.png"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mask_{timestamp}.png"

    mask_path = output_path_base / filename

    # 7. 保存掩码
    try:
        mask_path.write_bytes(resp.content)
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to save mask file: {e}",
            "mask_path": None,
        }

    # 8. Build response
    result: dict[str, Any] = {
        "success": True,
        "message": f"Segmentation completed, mask saved to {mask_path}",
        "mask_path": str(mask_path),
        "mask_data_url": _build_image_data_url_from_bytes(resp.content),
    }

    # Content blocks for the LLM: text + mask image
    content_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"MedSAM segmentation completed:\n- Status: Success\n- Mask saved at: {str(mask_path)}\n- Here is the segmentation mask image:"
        },
        {
            "type": "image_url",
            "image_url": {"url": result["mask_data_url"]}
        }
    ]

    return content_blocks, result

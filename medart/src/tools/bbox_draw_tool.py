"""
Bbox drawing tool for LangChain agent — draws rectangle(s) on an image from
normalized coordinates.

提供"按归一化坐标在图像上画矩形框"的能力。在本项目的医学影像 claim
验证场景中，可用于让 LLM 显式标出它关注的解剖结构/病灶区域，并把标注后
的图像回传给多模态 LLM 复核（例如确认测量位置、侧别、病灶范围）。

遵循 image_flip_tool.py 的模式：
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组
  - content_blocks 包含标注后图像的 data URL，让多模态 LLM 直接看到
  - result 为应用层元数据（路径、尺寸、像素坐标、状态）
- 结果持久化到本地目录

坐标约定：
- 每个框为 [x1, y1, x2, y2]，取值范围 [0, 1]，原点为图像左上角，
  x 向右、y 向下。x1/y1 为左上角点，x2/y2 为右下角点。
- 越界值会被裁剪到 [0, 1]；x1>x2 / y1>y2 会自动交换；两种情况都会在
  result 的 warnings 中记录。

结果持久化（本地保存，辅助调试 / 审计）：
- bbox_drawn/{timestamp}_{name}_boxed.png   标注后的图像文件（PNG 无损）
"""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.tools import tool
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT
from src.multi_agent.image_utils import build_image_data_url


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class BoxSpec(BaseModel):
    """单个矩形框：归一化坐标 + 可选标签 / 颜色。"""

    box: list[float] = Field(
        description=(
            "Normalized rectangle [x1, y1, x2, y2] in range [0, 1]. "
            "Origin is the top-left corner of the image; x grows to the "
            "right, y grows downward. (x1, y1) is the top-left corner and "
            "(x2, y2) is the bottom-right corner of the rectangle."
        ),
        min_length=4,
        max_length=4,
    )
    label: str | None = Field(
        default=None,
        description=(
            "Optional text label drawn next to the rectangle "
            "(e.g. the structure name). Omit for no label."
        ),
    )
    color: str | None = Field(
        default=None,
        description=(
            "Optional rectangle color as a CSS name ('red', 'lime', ...) or "
            "hex string ('#FF0000'). Omit to use an automatic color cycle."
        ),
    )


class BboxDrawInput(BaseModel):
    image_path: str = Field(
        description="Input: Absolute path to the local image to annotate.",
    )
    boxes: list[BoxSpec] = Field(
        description=(
            "Input: One or more rectangles to draw, each with normalized "
            "[x1, y1, x2, y2] coordinates in [0, 1] plus an optional label "
            "and color."
        ),
        min_length=1,
    )
    line_width: int | None = Field(
        default=None,
        description=(
            "Optional rectangle border width in pixels. Omit (recommended) "
            "to auto-scale with the image size."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional full file path (including file name) where the "
            "annotated image should be saved. Omit to auto-generate a "
            "timestamped file name under the default output directory."
        ),
    )


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# 自动配色循环（框未指定颜色时依次取用）
_DEFAULT_COLORS = ["red", "lime", "blue", "orange", "magenta", "cyan", "yellow"]

_FONT_CANDIDATES = ["arial.ttf", "arialbd.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf"]


def _ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pick_font(size: int) -> ImageFont.ImageFont:
    """尽量取一个指定大小的 TrueType 字体，失败则退回默认位图字体。"""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow ≥ 10.1
    except TypeError:
        return ImageFont.load_default()


def _coerce_box_spec(spec: Any) -> dict[str, Any]:
    """统一单个框输入的形态为 dict。

    - dict：原样返回；
    - Pydantic 模型（langchain 工具经 args_schema 校验后传入 BoxSpec）：
      转成 dict（兼容 v2 的 model_dump 与 v1 的 dict）；
    - 其他对象：按属性提取 box/label/color。
    """
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, BaseModel):
        if hasattr(spec, "model_dump"):  # Pydantic v2
            return spec.model_dump()
        return spec.dict()  # Pydantic v1
    return {
        "box": getattr(spec, "box", []),
        "label": getattr(spec, "label", None),
        "color": getattr(spec, "color", None),
    }


def _resolve_color(color: str | None, fallback: str) -> str:
    """校验颜色字符串；非法或为空时回退到给定默认色。"""
    if not color:
        return fallback
    try:
        ImageColor.getrgb(color)
        return color
    except Exception:
        return fallback


def _text_contrast_color(bg_color: str) -> str:
    """根据背景色亮度选择黑/白文字色，保证标签可读。"""
    try:
        r, g, b = ImageColor.getrgb(bg_color)[:3]
    except Exception:
        return "black"
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 140 else "white"


def _normalize_box(
    raw_box: list[float],
    index: int,
    warnings: list[str],
) -> tuple[float, float, float, float] | None:
    """校验并规整单个归一化框，返回 (x1, y1, x2, y2)；无效时返回 None。"""
    if len(raw_box) != 4:
        warnings.append(f"Box #{index}: expected 4 coordinates, got {len(raw_box)}; skipped.")
        return None

    x1, y1, x2, y2 = (float(v) for v in raw_box)

    # 越界裁剪
    clamped = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
    if clamped != [x1, y1, x2, y2]:
        warnings.append(
            f"Box #{index}: coordinates out of [0, 1], clamped "
            f"{[x1, y1, x2, y2]} → {clamped}."
        )
    x1, y1, x2, y2 = clamped

    # 角点顺序交换
    if x1 > x2:
        x1, x2 = x2, x1
        warnings.append(f"Box #{index}: x1 > x2, swapped.")
    if y1 > y2:
        y1, y2 = y2, y1
        warnings.append(f"Box #{index}: y1 > y2, swapped.")

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Pure drawing function (pipeline-callable, no langchain dependency)
# ---------------------------------------------------------------------------


def draw_normalized_boxes(
    src_path: str,
    boxes: list[dict[str, Any]],
    out_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    line_width: int | None = None,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """在图像上按归一化坐标绘制矩形框并保存，返回 (输出路径, 像素框信息, 警告)。

    纯函数、不依赖 langchain，既可被 @tool 包装给 LLM 调用，也可被流水线
    程序直接调用。

    Parameters
    ----------
    src_path : str
        源图像路径。
    boxes : list[dict]
        每个元素形如 {"box": [x1, y1, x2, y2], "label": str|None,
        "color": str|None}，坐标归一化到 [0, 1]。
    out_path : str | Path | None
        完整输出路径（含文件名）。若提供则直接使用，忽略 out_dir。
    out_dir : str | Path | None
        输出目录。未提供且未提供 out_path 时，输出到源图像同目录。
    line_width : int | None
        框线宽度（像素）。None 时按图像尺寸自动缩放。

    Returns
    -------
    (Path, list[dict], list[str])
        保存路径；每个已绘制框的像素坐标信息；处理过程中的警告列表。

    Raises
    ------
    FileNotFoundError  源文件不存在。
    ValueError         没有任何有效的框可绘制。
    Exception          读取 / 绘制 / 保存失败时向上抛出。
    """
    src = Path(src_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Image file not found: '{src_path}'")

    with Image.open(src) as img:
        img.load()
        width, height = img.size
        # 统一转到 RGB/RGBA 以便绘制与保存（调色板等模式不能直接画）
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        canvas = img.copy()

    warnings: list[str] = []
    drawn: list[dict[str, Any]] = []

    # 自动线宽：按图像最长边缩放
    if line_width is None:
        effective_line_width = max(2, round(max(width, height) / 300))
    else:
        effective_line_width = max(1, int(line_width))

    # 标签字体大小随图像缩放
    font_size = max(12, round(max(width, height) / 40))
    font = _pick_font(font_size)

    draw = ImageDraw.Draw(canvas)

    for idx, raw_spec in enumerate(boxes):
        spec = _coerce_box_spec(raw_spec)
        raw_box = spec.get("box", [])
        normalized = _normalize_box(raw_box, idx + 1, warnings)
        if normalized is None:
            continue
        x1, y1, x2, y2 = normalized

        # 归一化 → 像素
        px1 = round(x1 * (width - 1))
        py1 = round(y1 * (height - 1))
        px2 = round(x2 * (width - 1))
        py2 = round(y2 * (height - 1))

        # 退化框（转换后面积为 0）跳过
        if px2 - px1 < 1 or py2 - py1 < 1:
            warnings.append(f"Box #{idx + 1}: degenerate (zero area after pixel conversion); skipped.")
            continue

        color = _resolve_color(spec.get("color"), _DEFAULT_COLORS[len(drawn) % len(_DEFAULT_COLORS)])
        draw.rectangle([px1, py1, px2, py2], outline=color, width=effective_line_width)

        label = (spec.get("label") or "").strip()
        # if label:
        #     # 文字背景框：优先画在原框上方，空间不够则画在框内顶部
        #     tb = draw.textbbox((0, 0), label, font=font)
        #     tw, th = tb[2] - tb[0], tb[3] - tb[1]
        #     pad = max(2, effective_line_width // 2)
        #     ty = py1 - th - 2 * pad
        #     if ty < 0:
        #         ty = py1 + pad
        #     tx = px1
        #     draw.rectangle(
        #         [tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad],
        #         fill=color,
        #     )
        #     draw.text((tx + pad, ty + pad), label, fill=_text_contrast_color(color), font=font)

        drawn.append({
            "index": idx + 1,
            "normalized_box": [x1, y1, x2, y2],
            "pixel_box": [px1, py1, px2, py2],
            "label": label or None,
            "color": color,
        })

    if not drawn:
        raise ValueError("No valid boxes to draw (all provided boxes were invalid or degenerate).")

    # 决定输出路径（标注图统一存 PNG，避免 JPEG 压缩污染框线/文字）
    if out_path is not None:
        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(out_dir) if out_dir is not None else src.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = base_dir / f"{timestamp}_{src.stem}_boxed.png"

    canvas.save(save_path)
    return save_path, drawn, warnings


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=BboxDrawInput)
def draw_bbox_on_image(
    image_path: str,
    boxes: list[dict[str, Any]],
    line_width: int | None = None,
    output_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw one or more rectangles on an image using NORMALIZED coordinates.

    Use this tool to mark the exact region(s) you are reasoning about — for
    example the anatomical structure, lesion, or measurement site behind a
    claim. The annotated image is returned so you can directly view and
    re-verify your markings.

    Coordinate convention:
    - Each box is [x1, y1, x2, y2] normalized to [0, 1], origin at the
      TOP-LEFT corner, x to the right, y downward.
    - Out-of-range values are clamped; swapped corners are auto-corrected
      (both reported in warnings).

    Inputs:
    - image_path: Absolute path to the local image file to annotate
    - boxes: List of rectangles, each {"box": [x1, y1, x2, y2],
      "label": optional text, "color": optional CSS color or #RRGGBB}
    - line_width: Optional border width in pixels (omit to auto-scale)
    - output_path: Optional full file path (including file name) to save the
      annotated image (omit for an auto-generated timestamped path)

    Outputs (content_blocks → visible to LLM):
    - Text: Drawing summary (paths, dimensions, per-box pixel coordinates)
    - Image: The annotated image as a data URL

    Outputs (result dict → application metadata):
    - success: Whether the drawing succeeded
    - input_path / output_path: Source and annotated image paths
    - width / height: Image dimensions in pixels
    - boxes_drawn: Per-box info (normalized + pixel coordinates, label, color)
    - warnings: Non-fatal issues (clamping, swapping, skipped boxes)
    - message: Status description or error message
    """

    def _error_return(msg: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"Bbox drawing failed: {msg}"}
        ]
        result: dict[str, Any] = {
            "success": False,
            "input_path": image_path,
            "output_path": "",
            "width": 0,
            "height": 0,
            "boxes_drawn": [],
            "warnings": [],
            "message": msg,
        }
        return content_blocks, result

    # =====================================================================
    # 1. 校验输入
    # =====================================================================
    src = Path(image_path)
    if not image_path or not src.exists():
        return _error_return(f"Image file not found: '{image_path}'")
    if not src.is_file():
        return _error_return(f"Path is not a file: '{image_path}'")
    if not boxes:
        return _error_return("No boxes provided; pass at least one rectangle.")

    try:
        with Image.open(src) as img:
            width, height = img.size
    except Exception as e:
        traceback.print_exc()
        return _error_return(f"Failed to open image '{image_path}': {e}")

    # =====================================================================
    # 2. 绘制并保存（交由共享纯函数处理）
    # =====================================================================
    try:
        if output_path:
            save_path, drawn, warnings = draw_normalized_boxes(
                image_path,
                boxes,
                out_path=output_path,
                line_width=line_width,
            )
        else:
            out_dir = _ensure_dir(PROJECT_ROOT / "bbox_drawn")
            save_path, drawn, warnings = draw_normalized_boxes(
                image_path,
                boxes,
                out_dir=out_dir,
                line_width=line_width,
            )
    except Exception as e:
        traceback.print_exc()
        return _error_return(f"Failed to draw boxes on '{image_path}': {e}")

    # =====================================================================
    # 3. 构建 data URL 供 LLM 查看
    # =====================================================================
    try:
        data_url = build_image_data_url(str(save_path))
    except Exception as e:
        traceback.print_exc()
        return _error_return(
            f"Boxes drawn and saved but failed to encode data URL: {e}"
        )

    # =====================================================================
    # 4. 构建 LLM content blocks（文本 + 标注图像）
    # =====================================================================
    box_lines = []
    for info in drawn:
        nb = info["normalized_box"]
        pb = info["pixel_box"]
        label_part = f" '{info['label']}'" if info["label"] else ""
        box_lines.append(
            f"- Box {info['index']}{label_part} [{info['color']}]: "
            f"normalized [{nb[0]:.3f}, {nb[1]:.3f}, {nb[2]:.3f}, {nb[3]:.3f}] → "
            f"pixel [{pb[0]}, {pb[1]}, {pb[2]}, {pb[3]}]"
        )

    summary_parts = [
        f"## Rectangles Drawn on Image\n",
        f"**Input**: {image_path}",
        f"**Output**: {save_path}",
        f"**Dimensions**: {width}×{height}px",
        f"**Boxes drawn**: {len(drawn)} of {len(boxes)} requested",
        "",
        *box_lines,
    ]
    if warnings:
        summary_parts += ["", "**Warnings**:", *[f"- {w}" for w in warnings]]
    summary_parts.append(
        "\nThe following is the annotated image with the rectangles drawn on it:\n"
    )

    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(summary_parts)},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    # =====================================================================
    # 5. 构建 result dict
    # =====================================================================
    result_dict: dict[str, Any] = {
        "success": True,
        "input_path": str(src),
        "output_path": str(save_path),
        "width": width,
        "height": height,
        "boxes_drawn": drawn,
        "warnings": warnings,
        "message": (
            f"Drew {len(drawn)} box(es) on '{image_path}' ({width}×{height}px), "
            f"saved to {save_path}"
        ),
    }

    return content_blocks, result_dict

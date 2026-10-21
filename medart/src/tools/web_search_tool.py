"""
Web search tool using Doubao (豆包) Global Search API for LangChain agent.

Provides web search capabilities to the agent. Calls the Doubao Global Search
API (https://open.feedcoopapi.com/search_api/global_search) and returns
relevant document snippets, titles, and URLs along with downloaded images
(both as data URLs visible to the LLM and as local files).

结果持久化（本地保存，辅助调试 / 审计）：
- 原始 API 响应 JSON → search/{timestamp}/raw_response.json
- 格式化结果 JSON     → search/{timestamp}/results.json
- 下载的图片           → search/{timestamp}/img/rank_{N}_{idx}.{ext}

返回给 LLM 的内容块包含文本 + 图片 data URL，让大模型能直接看到图片。

遵循 medsam_tool.py / biomedparse_client.py 的模式：
- @tool + Pydantic args_schema
- HTTP 调用 + 完整错误处理
- 返回 (content_blocks, result) 元组
- 结果保存到本地目录

Authentication: Bearer token via environment variable DOUBAO_SEARCH_API_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.config import DOUBAO_SEARCH_URL, PROJECT_ROOT,DOUBAO_SEARCH_KYE


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Input: Search query string, 1–100 characters. "
            "Supports multi-language queries."
        ),
    )
    doc_count: int | None = Field(
        default=10,
        description=(
            "Input: Number of search results to return (max 20). "
            "Default is 10."
        ),
    )
    max_snippet_length: int | None = Field(
        default=1000,
        description=(
            "Input: Maximum snippet tokens per result (max 3000). "
            "Default 1000. Recommended 1000."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    """将字符串中的非法文件名字符替换为下划线，并截断到 max_len。"""
    safe = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe = re.sub(r"\s+", "_", safe)
    return safe.strip("_")[:max_len]


def _ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_data_url(png_bytes: bytes, mime: str = "image/png") -> str:
    """将图片字节转换为 data URL，供 LLM content block 使用。"""
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _infer_mime(image_url: str) -> str:
    """从图片 URL 推断 MIME 类型。"""
    url_lower = image_url.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    for ext, mime in mapping.items():
        if ext in url_lower:
            return mime
    return "image/jpeg"


def _download_image(
    image_url: str,
    save_dir: Path,
    rank: int,
    img_idx: int,
    timeout: int = 15,
) -> dict[str, Any]:
    """
    下载单张图片并保存到 save_dir。

    Returns
    -------
    dict 包含 local_path, file_name, data_url, success, message。
    """
    ext = ".jpg"
    for known_ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]:
        if known_ext in image_url.lower():
            ext = known_ext
            break

    file_name = f"rank_{rank}_{img_idx}{ext}"
    local_path = save_dir / file_name
    mime = _infer_mime(image_url)

    try:
        resp = requests.get(image_url, timeout=timeout)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        data_url = _build_data_url(resp.content, mime=mime)
        return {
            "local_path": str(local_path),
            "file_name": file_name,
            "data_url": data_url,
            "mime": mime,
            "success": True,
            "message": "",
        }
    except Exception as e:
        return {
            "local_path": "",
            "file_name": file_name,
            "data_url": "",
            "mime": mime,
            "success": False,
            "message": str(e),
        }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=WebSearchInput)
def web_search(
    query: str,
    doc_count: int | None = 10,
    max_snippet_length: int | None = 1000,
) -> tuple[Any, dict[str, Any]]:
    """Search the web using the Doubao Global Search API.

    This data is public. Do not reverse-engineer the source or use image IDs or similar metadata to deduce the correct answer.

    Performs a real-time web search and returns relevant results including
    title, URL, source, and content snippet for each match. Useful for
    looking up medical knowledge, literature, terminology, clinical guidelines,
    or any information not contained in the input data.
    Search results are for reference only; you should consider whether the search results are accurate.



    Inputs:
    - query: Search query string, 1-100 characters. Supports multi-language.
    - doc_count: Number of results to return (max 20). Default 10.
    - max_snippet_length: Maximum snippet tokens (max 3000). Default 1000.

    Outputs (content_blocks → visible to LLM):
    - Text: Summary + each result's title, URL, snippet
    - Images: Downloaded images as data URLs (if images were returned)

    Outputs (result dict → application metadata):
    - success: Whether the search succeeded
    - results: List of search result documents
    - images: List of downloaded image info with local_path and data_url
    - save_dir: Directory where results and images were saved
    - img_dir: Directory where images were saved
    - total_count: Total matching document count reported by API
    - message: Status description or error message
    """

    def _error_return(msg: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"Web search failed: {msg}"}
        ]
        result: dict[str, Any] = {
            "success": False,
            "results": [],
            "images": [],
            "save_dir": "",
            "img_dir": "",
            "total_count": 0,
            "message": msg,
        }
        return content_blocks, result

    # 1. Get API key (environment variable)
    api_key = DOUBAO_SEARCH_KYE
    if not api_key:
        return _error_return(
            "DOUBAO_SEARCH_API_KEY is not set. "
            "Please set this environment variable to use web search."
        )

    # 2. Build request
    url = DOUBAO_SEARCH_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"Query": query}
    if doc_count is not None:
        payload["DocCount"] = min(int(doc_count), 20)
    if max_snippet_length is not None:
        payload["MaxSnippetLength"] = min(int(max_snippet_length), 3000)

    # 3. Call API
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return _error_return("Web search request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        return _error_return(
            f"Cannot connect to Doubao Search API at {url}, "
            "please verify network connectivity."
        )
    except requests.exceptions.RequestException as e:
        return _error_return(f"Web search request failed: {e}")

    # 4. Parse response JSON
    try:
        data: dict[str, Any] = resp.json()
    except json.JSONDecodeError:
        return _error_return("Failed to parse search API JSON response.")

    # 5. Check for API-level error (ResponseMetadata.Error)
    response_metadata = data.get("ResponseMetadata", {})
    api_error = response_metadata.get("Error")
    if api_error:
        error_msg = api_error.get("Message", "Unknown API error")
        error_code = api_error.get("CodeN", api_error.get("Code", "unknown"))
        return _error_return(f"Search API error [{error_code}]: {error_msg}")

    # 6. Check Result field
    result = data.get("Result")
    if result is None:
        return _error_return("Search API returned null Result.")

    # 7. Check result-level error code
    error_code = result.get("ErrorCode", 0)
    error_msg = result.get("ErrorMsg", "")
    if error_code != 0:
        return _error_return(f"Search result error [{error_code}]: {error_msg}")

    # =====================================================================
    # 8. 准备输出目录
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = _sanitize_filename(query)
    run_dir = _ensure_dir(PROJECT_ROOT / "search" / f"{timestamp}_{query_slug}")
    img_dir = _ensure_dir(run_dir / "img")

    # 8a. 保存原始 API 响应（非致命）
    raw_response_path = run_dir / "raw_response.json"
    try:
        raw_response_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # =====================================================================
    # 9. 提取文档 & 下载图片
    # =====================================================================
    documents = result.get("Documents", [])
    total_count = result.get("TotalDocCount", 0)

    formatted_results: list[dict[str, Any]] = []
    all_downloaded_images: list[dict[str, Any]] = []

    for doc in documents:
        rank = doc.get("Rank", 0)
        title = doc.get("Title", "").strip()
        doc_url = doc.get("Url", "").strip()
        host_info = doc.get("HostInfo", {})
        hostname = host_info.get("Hostname", "").strip()
        doc_info = doc.get("DocumentInfo", {})
        content_chars = doc_info.get("ContentCharCount", 0)

        snippets = doc.get("Snippet", [])
        text_parts: list[str] = []
        image_infos: list[dict[str, Any]] = []
        img_idx = 0

        for snippet in snippets:
            snippet_type = snippet.get("Type")
            if snippet_type == "text":
                text = snippet.get("Text", "").strip()
                if text:
                    text_parts.append(text)
            elif snippet_type == "image":
                image_data = snippet.get("Image", {})
                image_url = (image_data or {}).get("ImageUrl", "")
                if image_url:
                    dl_result = _download_image(
                        image_url, img_dir, rank, img_idx
                    )
                    all_downloaded_images.append({
                        **dl_result,
                        "rank": rank,
                        "image_url": image_url,
                        "width": image_data.get("Width"),
                        "height": image_data.get("Height"),
                    })
                    image_infos.append({
                        "rank": rank,
                        "index": img_idx,
                        "source_url": image_url,
                        "local_path": dl_result["local_path"],
                        "data_url": dl_result["data_url"],
                        "downloaded": dl_result["success"],
                    })
                    img_idx += 1

        combined_snippet = "\n".join(text_parts)

        formatted_results.append({
            "rank": rank,
            "title": title,
            "url": doc_url,
            "source": hostname,
            "content_length": content_chars,
            "snippet": combined_snippet,
            "images": image_infos,
        })

    # =====================================================================
    # 10. 保存格式化结果到 JSON（非致命）
    # =====================================================================
    output_data = {
        "query": query,
        "timestamp": timestamp,
        "total_count": total_count,
        "result_count": len(formatted_results),
        "image_count": len(all_downloaded_images),
        "results": formatted_results,
        "images": all_downloaded_images,
    }
    results_path = run_dir / "results.json"
    try:
        results_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # =====================================================================
    # 11. 构建 LLM content blocks（文本 + 图片 data URL）
    # =====================================================================
    content_blocks: list[dict[str, Any]] = []

    # 11a. 概要文本
    content_blocks.append({
        "type": "text",
        "text": (
            f"## Web Search Results\n\n"
            f"**Query**: {query}\n"
            f"**Results**: {len(formatted_results)} returned (total matches: {total_count})\n"
            f"**Images downloaded**: {len(all_downloaded_images)}\n\n"
            f"---\n"
        ),
    })

    # 11b. 每个文档的文本摘要
    for res in formatted_results:
        snippet_preview = res["snippet"][:500] if res["snippet"] else "(no snippet)"
        content_blocks.append({
            "type": "text",
            "text": (
                f"### [{res['rank']}] {res['title']}\n"
                f"**URL**: {res['url']}\n"
                f"**Source**: {res['source']}\n"
                f"**Content length**: {res['content_length']} chars\n\n"
                f"{snippet_preview}\n\n"
                f"---\n"
            ),
        })

    # 11c. 图片（如果有的话）
    if all_downloaded_images:
        content_blocks.append({
            "type": "text",
            "text": (
                f"### Images from Search Results\n\n"
                f"The following {len(all_downloaded_images)} image(s) were returned "
                f"along with the search results:\n"
            ),
        })

        for img_info in all_downloaded_images:
            if img_info["success"] and img_info.get("data_url"):
                caption = (
                    f"Rank {img_info['rank']}, image {img_info['file_name']}"
                )
                if img_info.get("width") and img_info.get("height"):
                    caption += f" ({img_info['width']}×{img_info['height']}px)"
                content_blocks.append({
                    "type": "text",
                    "text": f"**{caption}**\n",
                })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": img_info["data_url"]},
                })
    else:
        content_blocks.append({
            "type": "text",
            "text": "(No images were returned with these search results.)",
        })

    # =====================================================================
    # 12. 构建 result dict
    # =====================================================================
    result_dict: dict[str, Any] = {
        "success": True,
        "results": formatted_results,
        "images": all_downloaded_images,
        "save_dir": str(run_dir),
        "img_dir": str(img_dir),
        "total_count": total_count,
        "message": (
            f"Found {len(formatted_results)} result(s) "
            f"(total matches: {total_count}), "
            f"downloaded {len(all_downloaded_images)} image(s). "
            f"Saved to {run_dir}/"
        ),
    }

    return content_blocks, result_dict

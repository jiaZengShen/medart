"""
Web fetch tool for LangChain agent — fetches full page content from a URL.

Provides the ability to read detailed web page content, complementing the
web_search tool which only returns snippets/previews. After searching and
finding relevant URLs, the LLM can use this tool to read the full page.

遵循 web_search_tool.py / biomedparse_client.py 的模式：
- @tool + Pydantic args_schema
- HTTP 调用 + 完整错误处理
- 返回 (content_blocks, result) 元组
- 结果持久化到本地目录（原始 HTML + 提取文本 + 元数据）

结果持久化（本地保存，辅助调试 / 审计）：
- fetch/{timestamp}_{slug}/
  - raw_response.html    原始 HTML（UTF-8 重编码保存）
  - extracted.txt        提取的可读纯文本
  - metadata.json        请求元数据（URL、标题、长度、状态码等）
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class WebFetchInput(BaseModel):
    url: str = Field(
        description="The URL of the web page to fetch and read.",
    )
    max_chars: int | None = Field(
        default=10000,
        description=(
            "Maximum number of characters to return from the extracted text. "
            "Default 10000. Use larger values (e.g. 50000) for long documents."
        ),
    )
    timeout: int | None = Field(
        default=30,
        description="Request timeout in seconds. Default 30.",
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


def _extract_readable_text(html: str) -> str:
    """使用 BeautifulSoup 从 HTML 中提取可读文本内容。

    策略
    ----
    1. 完全移除 script / style / nav / footer / header / iframe 等非内容元素
    2. 优先从 <p> / <h1-6> / <li> / <blockquote> / <pre> 等语义标签提取，
       并过滤 < 10 字符的碎片（通常是导航/装饰文本）
    3. 如果语义提取结果 < 100 字符（页面结构特殊），回退到 <body> 级别提取
    4. 压缩多余空白行
    """
    soup = BeautifulSoup(html, "html.parser")

    # 移除不需要的元素
    for tag in soup(
        [
            "script", "style", "nav", "footer", "header",
            "noscript", "iframe", "form", "button", "input",
            "svg", "canvas", "aside", "select", "option",
        ]
    ):
        tag.decompose()

    # 策略 A：从语义标签提取——保持段落结构
    content_tags = soup.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6",
         "li", "blockquote", "pre", "td", "th",
         "div", "section", "article", "main"]
    )
    lines: list[str] = []
    for element in content_tags:
        text = element.get_text(strip=True)
        if text and 10 < len(text) < 5000:  # 过滤导航碎片和异常大块
            lines.append(text)

    extracted = "\n\n".join(lines)
    extracted = re.sub(r"\n{3,}", "\n\n", extracted)

    # 如果策略 A 提取太少，退回策略 B
    if len(extracted) < 100:
        body = soup.find("body")
        if body:
            extracted = body.get_text(separator="\n", strip=True)
        else:
            extracted = soup.get_text(separator="\n", strip=True)
        extracted = re.sub(r"\n{3,}", "\n\n", extracted)

    return extracted.strip()


def _make_session() -> requests.Session:
    """创建一个带合理请求头的 Session，降低被拦截的概率。"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    return session


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=WebFetchInput)
def web_fetch(
    url: str,
    max_chars: int | None = 10000,
    timeout: int | None = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch and read the full content of a web page.

    Use this tool when you need to read the detailed content of a web page,
    beyond what the web_search tool's snippet provides. For example, after
    searching and finding relevant URLs, call this tool to read the full
    article, guideline, or document.

    This is useful for:
    - Reading full medical articles or clinical guidelines
    - Getting detailed information from a specific web page
    - Verifying claims against source documents
    - Extracting specific data points from a known URL

    Inputs:
    - url: The URL of the web page to fetch
    - max_chars: Maximum characters to return (default 10000)
    - timeout: Request timeout in seconds (default 30)

    Outputs (content_blocks → visible to LLM):
    - Text: Page title + extracted content (with URL and metadata)

    Outputs (result dict → application metadata):
    - success: Whether the fetch succeeded
    - url: The requested URL
    - title: Page title (if extracted)
    - content_length: Length of extracted text
    - truncated: Whether content was truncated due to max_chars
    - save_dir: Directory where raw data was saved
    - message: Status description or error message
    """

    def _error_return(msg: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"Web fetch failed: {msg}"}
        ]
        result: dict[str, Any] = {
            "success": False,
            "url": url,
            "title": "",
            "content_length": 0,
            "truncated": False,
            "save_dir": "",
            "message": msg,
        }
        return content_blocks, result

    # =====================================================================
    # 1. 校验 URL
    # =====================================================================
    if not url or not url.startswith(("http://", "https://")):
        return _error_return(
            f"Invalid URL: '{url}'. URL must start with http:// or https://."
        )

    # =====================================================================
    # 2. 发起 HTTP 请求
    # =====================================================================
    try:
        session = _make_session()
        resp = session.get(
            url,
            timeout=min(int(timeout or 30), 120),
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return _error_return(
            f"Request timed out after {timeout}s for URL: {url}"
        )
    except requests.exceptions.ConnectionError:
        return _error_return(
            f"Cannot connect to {url}, please verify the URL is correct "
            "and that the server is reachable."
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        return _error_return(
            f"HTTP {status} error fetching {url}: {e}"
        )
    except requests.exceptions.TooManyRedirects:
        return _error_return(
            f"Too many redirects for URL: {url}"
        )
    except requests.exceptions.RequestException as e:
        return _error_return(f"Request failed: {e}")

    # =====================================================================
    # 3. 处理编码
    # =====================================================================
    encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    try:
        html_content = resp.content.decode(encoding, errors="replace")
    except Exception:
        html_content = resp.text  # fallback: 使用 requests 自身的解码

    # =====================================================================
    # 4. 提取页面标题
    # =====================================================================
    title = ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
    except Exception:
        pass

    # =====================================================================
    # 5. 提取可读文本
    # =====================================================================
    extracted_text = _extract_readable_text(html_content)

    if not extracted_text.strip():
        return _error_return(
            f"Fetched {url} successfully but could not extract any readable "
            "text content. The page may require JavaScript to render."
        )

    # =====================================================================
    # 6. 按 max_chars 截断
    # =====================================================================
    max_c = int(max_chars or 10000)
    truncated = len(extracted_text) > max_c
    display_text = extracted_text[:max_c]
    if truncated:
        display_text += (
            f"\n\n[... 内容已截断，完整内容共 {len(extracted_text)} 字符，"
            f"仅显示前 {max_c} 字符。如需完整内容请调用 web_fetch 时"
            f"增大 max_chars={len(extracted_text)}]"
        )

    # =====================================================================
    # 7. 结果持久化
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    url_slug = _sanitize_filename(url)
    run_dir = _ensure_dir(PROJECT_ROOT / "fetch" / f"{timestamp}_{url_slug}")

    raw_path = run_dir / "raw_response.html"
    extracted_path = run_dir / "extracted.txt"
    metadata_path = run_dir / "metadata.json"

    try:
        raw_path.write_text(html_content, encoding="utf-8")
        extracted_path.write_text(extracted_text, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "timestamp": timestamp,
                    "title": title,
                    "status_code": resp.status_code,
                    "encoding": encoding,
                    "final_url": resp.url,
                    "content_length": len(extracted_text),
                    "truncated": truncated,
                    "max_chars": max_c,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    # =====================================================================
    # 8. 构建 LLM content blocks
    # =====================================================================
    content_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"## Web Page Content\n\n"
                f"**URL**: {url}\n"
                f"**Final URL**: {resp.url}\n"
                f"**Title**: {title or '(no title)'}\n"
                f"**Content length**: {len(extracted_text)} chars"
                f"{' (truncated to ' + str(max_c) + ')' if truncated else ''}\n\n"
                f"---\n\n"
                f"{display_text}"
            ),
        }
    ]

    # =====================================================================
    # 9. 构建 result dict
    # =====================================================================
    result_dict: dict[str, Any] = {
        "success": True,
        "url": url,
        "final_url": resp.url,
        "title": title,
        "content_length": len(extracted_text),
        "truncated": truncated,
        "save_dir": str(run_dir),
        "message": (
            f"Fetched {url}: {len(extracted_text)} chars extracted"
            f"{' (truncated to ' + str(max_c) + ')' if truncated else ''}, "
            f"saved to {run_dir}/"
        ),
    }

    return content_blocks, result_dict

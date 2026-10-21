"""扩展的 LLM 类，包含 RawChatOpenAI（带原始日志功能）。"""

import json
import re
from copy import deepcopy
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


def _repair_tool_arguments(arguments: str) -> tuple[str, bool]:
    """修复工具调用参数的小错误。"""
    if not isinstance(arguments, str):
        return arguments, False

    candidates: list[str] = []
    stripped = arguments.strip()
    if stripped:
        candidates.append(stripped)
        trimmed = stripped
        while trimmed.startswith("{}"):
            trimmed = trimmed[2:].lstrip()
            if trimmed:
                candidates.append(trimmed)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            repaired = json.dumps(parsed, ensure_ascii=False)
            return repaired, repaired != stripped

    return arguments, False


_IMAGE_DATA_URL_RE = re.compile(r"(data:image/[^;]+;base64,)([A-Za-z0-9+/=]+)")
_UNSUPPORTED_IMAGE_MARKER = "[Image could not be displayed: unsupported format]"


def _summarize_large_log_value(value):
    """在日志中简化大的 base64 图片值。"""
    if isinstance(value, dict):
        return {key: _summarize_large_log_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_summarize_large_log_value(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace_data_url(match: re.Match[str]) -> str:
        prefix, data = match.groups()
        return f"{prefix}[base64 omitted, {len(data)} chars]"

    summarized = _IMAGE_DATA_URL_RE.sub(replace_data_url, value)
    if summarized != value:
        return summarized

    marker_index = value.find(_UNSUPPORTED_IMAGE_MARKER)
    if marker_index == -1:
        return value

    summarized = value[: marker_index + len(_UNSUPPORTED_IMAGE_MARKER)]
    omitted_chars = len(value) - len(summarized)
    if omitted_chars > 0:
        summarized += f" [truncated {omitted_chars} chars]"
    return summarized


# ---------------------------------------------------------------------------
# 账号额度限制 (429 AccountQuotaExceeded) — 触发整体停止
# ---------------------------------------------------------------------------
# 火山方舟账号有 5 小时滚动用量额度, 耗尽后所有请求都返回
#   RateLimitError 429 / code=AccountQuotaExceeded,
#   报错信息里带重置时间: "It will reset at 2026-08-14 00:50:54 +0800 CST".
# 等待/重试没有意义(额度按 5 小时窗口结算), 所以检测到额度耗尽时抛出
# QuotaExceededError, 让主程序整体停止, 由断点续传在额度重置后再跑。
#
# 注意: 另一种 429 是 AccountRateLimitExceeded(瞬时请求过频, 无重置时间),
# 两者用 _parse_quota_reset_time 区分 —— 只有带重置时间的 AccountQuotaExceeded
# 才会触发整体停止, 瞬时频率限制按普通错误交给上层处理。

class QuotaExceededError(RuntimeError):
    """账号 5 小时用量额度耗尽 (429 AccountQuotaExceeded)。

    抛出后应一路冒泡到主程序整体停止, 不要在中间 except 里吞掉。
    额度按 5 小时窗口结算, 等待无意义; 应停止后由断点续传在额度重置后再跑。
    """

    def __init__(self, message: str, reset_at=None):
        super().__init__(message)
        self.reset_at = reset_at


_QUOTA_RESET_RE = re.compile(r"reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})")
_QUOTA_RESET_FMT = "%Y-%m-%d %H:%M:%S %z"


def _is_account_quota_exceeded(error) -> bool:
    """判断错误是否为账号 5h 用量额度耗尽 (429 AccountQuotaExceeded)。

    注意与 AccountRateLimitExceeded(瞬时请求过频, 无重置时间)区分:
    只有 AccountQuotaExceeded 才触发整体停止。
    """
    # 1) openai SDK 顶层 code 属性
    if getattr(error, "code", None) == "AccountQuotaExceeded":
        return True
    # 2) body 字典里嵌套的 error.code
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("code") == "AccountQuotaExceeded":
            return True
        if body.get("code") == "AccountQuotaExceeded":
            return True
    # 3) 字符串兜底(不同 langchain 版本可能包成别的异常)
    if "AccountQuotaExceeded" in str(error):
        return True
    return False


def _parse_quota_reset_time(error) -> datetime | None:
    """从 429 AccountQuotaExceeded 错误里解析额度重置时间。

    返回 naive 判断:
    - 非 AccountQuotaExceeded 错误 → None
    - 是但解析不出时间 → None
    - 解析成功 → 带时区的 datetime
    """
    message = ""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            message = str(inner.get("message") or "")
    if not message:
        message = str(error)

    match = _QUOTA_RESET_RE.search(message)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), _QUOTA_RESET_FMT)
    except ValueError:
        return None


class RawChatOpenAI(ChatOpenAI):
    """扩展的 ChatOpenAI，记录原始的请求和响应到日志文件。

    特点：
    - 记录完整的请求和响应，便于调试
    - 在日志中自动省略大的 base64 图片
    - 自动修复工具调用参数中的小错误
    - 遇到账号额度耗尽 (429 AccountQuotaExceeded) 时自动等待重置后重试
    """

    def __init__(self, *args, log_file="llm_raw_log.txt", **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "log_file", log_file)

    def _append_log(self, label: str, payload):
        serializable = payload.model_dump() if hasattr(payload, "model_dump") else payload
        serializable = _summarize_large_log_value(serializable)
        log_text = f"\n{'='*20} {datetime.now()} {label} {'='*20}\n"
        log_text += json.dumps(serializable, ensure_ascii=False, indent=2)
        log_text += f"\n{'='*80}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_text)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        self._append_log("REQUEST", payload)
        return payload

    def _sanitize_response_tool_arguments(self, response_payload: dict) -> tuple[dict, list[dict[str, str | int]]]:
        sanitized = deepcopy(response_payload)
        repairs: list[dict[str, str | int]] = []

        for choice_index, choice in enumerate(sanitized.get("choices", [])):
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            for tool_index, tool_call in enumerate(tool_calls):
                function_payload = tool_call.get("function") or {}
                arguments = function_payload.get("arguments")
                repaired_arguments, changed = _repair_tool_arguments(arguments)
                if changed:
                    function_payload["arguments"] = repaired_arguments
                    repairs.append({
                        "choice_index": choice_index,
                        "tool_index": tool_index,
                        "tool_name": function_payload.get("name") or "",
                        "original_arguments": arguments,
                        "repaired_arguments": repaired_arguments,
                    })

        return sanitized, repairs

    def _create_chat_result(self, response, generation_info=None):
        response_payload = response.model_dump() if hasattr(response, "model_dump") else response
        self._append_log("RESPONSE", response_payload)
        sanitized_payload, repairs = self._sanitize_response_tool_arguments(response_payload)
        if repairs:
            self._append_log("RESPONSE_TOOL_ARGUMENT_REPAIRS", {"repairs": repairs})
        return super()._create_chat_result(response, generation_info)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """覆盖 _generate：账号 5h 额度耗尽 (429 AccountQuotaExceeded) 时抛出
        QuotaExceededError，让主程序整体停止。

        额度按 5 小时窗口结算，等待/重试没有意义，所以直接抛异常冒泡。
        普通错误(含瞬时频率限制 AccountRateLimitExceeded)原样抛出，
        由调用方(ReAct 循环 / 断点续传)自行处理。
        """
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            if not _is_account_quota_exceeded(exc):
                # 不是额度耗尽(或解析失败)的错误, 原样抛出
                raise
            reset_dt = _parse_quota_reset_time(exc)
            reset_txt = f"，将于 {reset_dt:%Y-%m-%d %H:%M:%S} 重置" if reset_dt else ""
            raise QuotaExceededError(
                f"账号 5h 用量额度耗尽 (AccountQuotaExceeded){reset_txt}。"
                f"程序已停止，请等待额度重置后使用 --resume 断点续传。",
                reset_at=reset_dt,
            ) from exc


class PrintRawCallback(BaseCallbackHandler):
    """回调处理器，打印原始的 LLM 响应到日志文件。"""

    def on_llm_end(self, response, raw=None, **kwargs):
        if raw is not None:
            log_text = f"\n{'='*20} {datetime.now()} CALLBACK RESPONSE {'='*20}\n"
            log_text += json.dumps(_summarize_large_log_value(raw), ensure_ascii=False, indent=2)
            log_text += f"\n{'='*80}\n"
            with open("llm_raw_callback.txt", "a", encoding="utf-8") as f:
                f.write(log_text)

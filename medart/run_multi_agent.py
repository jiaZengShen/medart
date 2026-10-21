#!/usr/bin/env python3
"""Entry point for the multi-agent inference pipeline.

Usage:
    python run_multi_agent.py --test-json ./input/recognition_export.json --limit 10

This mirrors the existing run_agent_4inference.py but uses the multi-agent
orchestrator instead of a single ReAct agent.
"""
from __future__ import annotations

import sys
import os



# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))






import argparse
import base64
import json
import mimetypes
import random
import re
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# Global lock for writing results JSON safely from parallel workers
_results_lock = threading.Lock()

from src.llms import QuotaExceededError, RawChatOpenAI as RawChatOpenAI
from src.corrector_agent import (
    CORRECTOR_BASE_URL,
    CORRECTOR_API_KEY,
    CORRECTOR_MODEL,
    CORRECTOR_MAX_ROUNDS,
    CORRECTOR_MAX_CONCURRENT,
    CORRECTOR_IMAGE_URL_PREFIX,
    CORRECTOR_IMAGE_PATH_DEPTH,
    ARBITER_ENABLED,
    ARBITER_MAX_RETRIES,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 压制无关噪音日志, 保持输出干净 (不影响功能)
# ---------------------------------------------------------------------------
import logging
import warnings

# 1) 每一条豆包 API 请求都打 INFO, 太吵, 只保留 WARNING 及以上
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 2) 多模态模型内部梯度检查点(推理时)的无害警告, 屏蔽
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*requires_grad.*")

from src.config import DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from src.eval.dataset_adapter import resolve_runtime_path
from src.multi_agent.orchestrator import Orchestrator
from src.single_agent_v1.agent_v2 import SingleAgentV2
from src.single_agent_v1.agent_v3 import SingleAgentV3
from src.single_agent_v1.agent_v4 import SingleAgentV4
from src.single_agent_v1.agent_v5 import MultiAgentV5
from src.single_agent_v1.agent_v6 import SingleAgentV6
from src.single_agent_v1.agent_v7 import SingleAgentV7, seed_fake_rng
from src.single_agent_v1.agent_v8 import SingleAgentV8
from src.corrector_agent.agent_base import SingleAgentV8_Base
from src.single_agent_v1.agent_v8_1 import SingleAgentV8_1
from src.single_agent_v1.agent_v8_2 import SingleAgentV8_2
from src.single_agent_v1.agent_v9 import SingleAgentV9
from src.single_agent_v1.debug_tracker import DebugTracker
from src.tools.biomedparse_client import biomedparse_segment
from src.tools.hc_measurement import measure_fetal_head_circumference
from src.tools.web_search_tool import web_search
from src.tools.web_fetch_tool import web_fetch
from src.tools.image_flip_tool import flip_image_horizontally
from src.tools.bbox_draw_tool import draw_bbox_on_image
from src.tools.multimodal_rag_tool import preload_rag_models, retrieve_medical_knowledge
# from src.tools.radimagenet_rag_tool import retrieve_radimagenet_knowledge
# from src.tools.miriad_rag_tool import retrieve_miriad_qa
from src.tools.unified_rag_tool import (
    retrieve_unified_knowledge,
    set_current_case_context,
    set_rag_call_log_file,
)
from src.tools.rule_rag_tool import retrieve_rule

# [EVIDENCE-FILTER] 让 RAG 证据过滤用题目 (claim) 而不是检索关键词判定
from src.tools.rag_evidence_filter import (
    set_claims_for_filter,
    set_filter_log_file,
)
from constant import COT_PROMPT


# ── Argument parsing (kept consistent with the existing entry point) ──
# ── Tools ──

AVAILABLE_TOOLS = [
    #measure_fetal_head_circumference,
    #web_search,
    #web_fetch,
    #flip_image_horizontally,
    #draw_bbox_on_image,

    # retrieve_medical_knowledge,
    # retrieve_miriad_qa,
    # retrieve_radimagenet_knowledge,
    retrieve_unified_knowledge,  #正确的的rag
    retrieve_rule, # 规则rag。

    #biomedparse_segment,
    #medsam_segment,
]

# ── LLM setup ──

def _build_llm(model_name: str = "doubao-seed-2.0-pro", log_file: str = "llm_raw_log.txt") -> RawChatOpenAI:
    """Create a ChatOpenAI instance (using RawChatOpenAI for logging).

    NOTE: You may want to swap the model / credentials or use a config
    file instead of hard-coding.  This mirrors the existing agent.py.
    """
    extra_body = {
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"}
    }

    return RawChatOpenAI(
        model=model_name,
            openai_api_key="******",
        base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
        temperature=0.1,
        top_p=0.9,
        log_file=log_file,
        # model_kwargs={
        #     "reasoningEffort": "high",
        #     "thinking": {"type": "enabled"}
        # },
        # 方法二：用 extra_body 包一下（更稳）
        #extra_body=extra_body
        # 长思维链测试: max_completion_tokens 计入思维链+回答, 设 16000 留足预算, 避免 finish_reason="length" 截断
        max_completion_tokens=16000,
    )



def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-agent pipeline on 4 inference types")
    parser.add_argument("--test-json", default="./input/recognition_export.json", help="Path to test JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results (默认: 新的时间戳目录; 使用 --resume 且未指定时, 自动定位最近的历史 outputs_* 目录)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling when using --limit (default: 42)")
    parser.add_argument("--inference-type", type=str, default=None, nargs='*',
                        help="Filter by inference type(s): recognition, parallel, measure, medical (can specify multiple)")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip image accessibility check")
    parser.add_argument("--replace-image-prefix", type=str, default="",
                        help="Replace image path prefix, format: old_prefix=new_prefix")
    parser.add_argument(
        "--server-data-root",
        type=str,
        default=None,
        help="Override runtime server data root used to map image paths",
    )
    parser.add_argument("--clear-llm-log", action="store_true",
                        help="Clear per-sample LLM log files before running (default: keep appending)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for sample execution (default: 1, fully serial)")
    # todo
    parser.add_argument("--agent-mode", type=str, default="single_v8", choices=["multi", "single", "single_v3", "single_v4", "single_v5", "single_v6", "single_v7"
        , "single_v8", "single_v8_1", "single_v8_2"
        , "single_v9", "single_v8_corrector"],
                        help="Agent mode: 'multi' for multi-agent orchestrator, 'single' for single-agent v2, 'single_v3' for single-agent v3 (two-stage), "
                             "'single_v4' for single-agent v4 (component-wise), 'single_v5' for multi-agent voting, 'single_v6' for single-agent v6 (three-stage, no OBVISER), "
                             "'single_v7' for single-agent v7 (question decomposition & merge), 'single_v8' for single-agent v8 (default: single_v3), "
                             "'single_v8_corrector' for v8 + corrector review loop")
    parser.add_argument("--triage-source", type=str, default="llm", choices=["llm"],
                        help="V7 triage source: 'llm' = LLM confidence triage (default);  experimental routing sampled from offline comparison_results4.csv labels")
    parser.add_argument("--comparison-csv", type=str, default="comparison_results4.csv",
                        help="Label CSV used by fake triage (default: ./comparison_results4.csv)")
    parser.add_argument("--fake-seed", type=int, default=42,
                        help="Random seed for the 70/30 & 50/50 sampling of fake triage (default: 42)")
    parser.add_argument("--debug-dir", type=str, default=str(DEFAULT_OUTPUT_DIR)+"/debug_reports",
                        help="Directory for debug HTML reports (single-agent mode only, default: <output_dir>/debug_reports)")
    parser.add_argument("--use-id-file", action="store_true",
                        help="If True, only validate samples whose IDs are in '待验证id.txt' file")
    parser.add_argument("--id-file", type=str, default="待验证id.txt",
                        help="Path to the ID file (default: 待验证id.txt)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 读取 checkpoint, 跳过上次已完成的样本, 重试失败/中断的样本, 在此基础上继续")
    parser.add_argument("--checkpoint-json", type=str, default=None,
                        help="checkpoint 文件路径 (默认 <output_dir>/checkpoint.json)")
    return parser.parse_args()


# ── Image helpers ──

def build_image_data_url(image_path: str) -> str:
    """Convert image file to data URL for LLM input."""
    image_file = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image_file.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def apply_server_data_root_override(samples: list[dict], server_data_root: str | None) -> None:
    if not server_data_root:
        return
    for sample in samples:
        image_paths = sample.get("image_paths")
        if isinstance(image_paths, list):
            sample["image_paths"] = [
                resolve_runtime_path(path, server_data_root=server_data_root)
                for path in image_paths
            ]


def verify_image_accessible(image_path: str) -> bool:
    """Check if image file exists and is readable."""
    from PIL import Image
    try:
        img = Image.open(image_path)
        img.convert("RGB")
        return True
    except Exception:
        traceback.print_exc()
        return False


def build_agent_prompt(sample: dict) -> str:
    """Build the agent-facing prompt from the local claim-type template."""
    claim_type = sample.get("claim_type", "recognition")
    if claim_type not in COT_PROMPT:
        raise ValueError(f"Unknown claim type: {claim_type}")

    prompt_template = COT_PROMPT[claim_type]
    return prompt_template.substitute(
        claim=sample.get("claim", ""),
        context=sample.get("context", "") or "",
    )


def _safe_filename_part(value: object) -> str:
    text = str(value).strip() if value is not None else "unknown"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unknown"


def _sample_log_path(log_dir: Path, sample: dict, sample_index: int) -> Path:
    claim_type = _safe_filename_part(sample.get("claim_type", "unknown"))
    sample_id = _safe_filename_part(sample.get("id", f"sample_{sample_index + 1}"))
    return log_dir / f"{claim_type}_{sample_id}_{sample_index + 1:04d}.txt"


def _run_single_sample(sample: dict, sample_index: int, log_dir: Path, agent_mode: str = "multi", debug_tracker: DebugTracker | None = None, triage_source: str = "llm", comparison_csv: str | None = None, fake_save_jsonl: str = "") -> dict:
    raw_claim = sample.get("claim", "")
    claim_type = sample.get("claim_type", "recognition")
    log_file = _sample_log_path(log_dir, sample, sample_index)

    user_input = {
        "case_id": sample.get("id", "unknown"),
        "claim": raw_claim,
        "agent_prompt": build_agent_prompt(sample),
        "claim_type": claim_type,
        "context": sample.get("context", ""),
        "image_paths": sample.get("image_paths", []),
        "true_label": sample.get("label", -1),
    }

    # [EVIDENCE-FILTER] 让 RAG 证据过滤用题目 (claim) 而不是检索关键词判定。
    # 注: orchestrator 用 ThreadPoolExecutor 派发子智能体, submit 时会复制
    # 当前 contextvar, 所以在此处(进入 agent 之前)注入即可被各线程继承。
    set_claims_for_filter(raw_claim)

    # [CALL-LOG] 注入当前题目上下文, 供 retrieve_unified_knowledge 记录"被哪个题目调用"
    set_current_case_context(
        sample_id=sample.get("id", "unknown"),
        claim=raw_claim,
        claim_type=claim_type,
    )

    if agent_mode == "single":
        # ── V2 单智能体模式 (三阶段+OBVISER) ──────────────────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV2(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            enable_reflection_loop=True,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v3":
        # ── V3 单智能体模式 (两阶段: 观察→判断) ─────────────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV3(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            judge_llm=None,  # uses same LLM for both stages
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v4":
        # ── V4 单智能体模式 (组件验证: 分解→填空→判断) ──────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV4(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            verifier_llm=None,  # uses same LLM for verification (swap for cheaper model if desired)
            enable_reflection_loop=True,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v5":
        # ── V5 多智能体投票模式 (多个独立智能体+投票) ───────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = MultiAgentV5(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            num_agents=5,
            debug_tracker=debug_tracker,
            log_file=str(log_file),
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v6":
        # ── V6 单智能体模式 (三阶段, 无OBVISER) ─────────────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV6(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v7":
        # ── V7 单智能体模式 (问题拆解-合并) ────────────────────
        # 拆解 claim → 逐题作答 → 合并判断（提示词偏见警告）
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV7(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
            triage_source=triage_source,
            comparison_csv=comparison_csv if triage_source == "fake" else None,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v8":
        # ── V8 单智能体模式 ──────────────────────────────────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV8(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
        # todo22
    elif agent_mode == "single_v8_2":
        # ── V8 单智能体模式 ──────────────────────────────────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV8_2(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v8_1":
        # ── V8_1 单智能体模式 (V8 基础上用于长思维链测试) ────────
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV8_1(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v9":
        # ── V9 统领 + 每选项/每判断验证子智能体 (多路径对抗 RAG) ──
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        agent = SingleAgentV9(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        result = agent.run(user_input)
    elif agent_mode == "single_v8_corrector":
        # ── V8 + 纠正者闭环（后置审查 + 反馈重答，最多 N 轮） ──
        from src.corrector_agent import CorrectorAgent
        llm = _build_llm(MODEL_NAME, log_file=str(log_file))
        base_agent = SingleAgentV8_Base(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            debug_tracker=debug_tracker,
        )
        agent = CorrectorAgent(
            llm=llm,
            tools=AVAILABLE_TOOLS,
            base_agent=base_agent,
            corrector_base_url=CORRECTOR_BASE_URL,
            corrector_api_key=CORRECTOR_API_KEY,
            corrector_model=CORRECTOR_MODEL,
            max_rounds=CORRECTOR_MAX_ROUNDS,
            max_concurrent=CORRECTOR_MAX_CONCURRENT,
            image_url_prefix=CORRECTOR_IMAGE_URL_PREFIX,
            image_path_depth=CORRECTOR_IMAGE_PATH_DEPTH,
            debug_tracker=debug_tracker,
            fake_save_jsonl=fake_save_jsonl,
            arbiter_enabled=ARBITER_ENABLED,
            arbiter_max_retries=ARBITER_MAX_RETRIES,
        )
        result = agent.run(user_input)
    else:
        # ── 多智能体模式（默认）────────────────────────────────────
        llm_planner = _build_llm(MODEL_NAME, log_file=str(log_file))
        llm_sub_agent = _build_llm(MODEL_NAME, log_file=str(log_file))
        llm_summarizer = _build_llm(MODEL_NAME, log_file=str(log_file))

        orchestrator = Orchestrator(
            llm_planner=llm_planner,
            llm_sub_agent=llm_sub_agent,
            llm_summarizer=llm_summarizer,
            tools=AVAILABLE_TOOLS,
        )
        result = orchestrator.run(user_input)

    # ── 填充公共字段 ────────────────────────────────────────
    result["sample_id"] = sample.get("id", "unknown")
    result["claim"] = raw_claim
    result["claim_type"] = claim_type
    result["true_label"] = sample.get("label", -1)
    result["log_file"] = str(log_file)
    result["agent_mode"] = agent_mode
    result['complete_time']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result["correct"] = _check_correctness(
        result["final_answer"],
        sample.get("label", -1),
        claim_type,
    )
    return result



def _run_single_sample_with_error_handling(sample: dict, sample_index: int, log_dir: Path, agent_mode: str = "multi", debug_tracker: DebugTracker | None = None, triage_source: str = "llm", comparison_csv: str | None = None, fake_save_jsonl: str = "") -> dict:
    try:
        return _run_single_sample(sample, sample_index, log_dir, agent_mode=agent_mode, debug_tracker=debug_tracker, triage_source=triage_source, comparison_csv=comparison_csv, fake_save_jsonl=fake_save_jsonl)
    except QuotaExceededError:
        raise  # 账号额度耗尽 → 冒泡到主程序整体停止
    except Exception as exc:
        traceback.print_exc()
        print(f"\nPipeline error on sample {sample.get('id', 'unknown')}: {exc}")
        return {
            "sample_id": sample.get("id", "unknown"),
            "claim": sample.get("claim", ""),
            "claim_type": sample.get("claim_type", "recognition"),
            "final_answer": "Uncertain",
            "confidence": 0.0,
            "status": "pipeline_error",
            "true_label": sample.get("label", -1),
            "correct": False,
            "latency_s": 0.0,
            "log_file": str(_sample_log_path(log_dir, sample, sample_index)),
            "agent_mode": agent_mode,
            "complete_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }



# ── Model assignment (all roles use the same model for now) ──
# All three roles (planner, sub-agent, summarizer) use the same model.
# You can swap to different models later for cost/quality tuning.

#MODEL_NAME = "doubao-seed-2.0-lite"
MODEL_NAME = "doubao-seed-2.0-pro"



# ── Stratified sampling ──────────────────────────────────────────────

def _stratified_sample_by_type(samples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample `n` items while preserving the claim_type proportion.

    If n < 4, just take the first n items (too small to stratify).
    If n >= 4, allocate each type its proportional share (at least 1 per
    type that exists in the data), then randomly sample within each type.
    Uses a fixed random seed for reproducibility.
    """
    # Set random seed for reproducible sampling
    rng = random.Random(seed)

    if n < 4 or not samples:
        return samples[:n]

    # Count types
    type_counts: Counter = Counter(s.get("claim_type", "unknown") for s in samples)
    total = len(samples)

    # Group by type
    type_groups: dict[str, list[dict]] = {}
    for s in samples:
        ct = s.get("claim_type", "unknown")
        type_groups.setdefault(ct, []).append(s)

    # Allocate slots: proportional, at least 1 per type that exists
    n_types = len(type_groups)
    allocation: dict[str, int] = {}
    remaining = n

    # First pass: give each type at least 1
    for ct in type_groups:
        allocation[ct] = 1
        remaining -= 1

    # Second pass: distribute remaining slots proportionally
    for ct in type_groups:
        if remaining > 0:
            extra = max(1, round(remaining * type_counts[ct] / total))
            extra = min(extra, remaining)  # don't exceed remaining
            # But cap at the actual number of samples available
            extra = min(extra, len(type_groups[ct]) - allocation[ct])
            allocation[ct] += extra
            remaining -= extra

    # If any slots still left, give them to the largest type
    if remaining > 0:
        largest_type = max(type_groups, key=lambda ct: len(type_groups[ct]))
        available = len(type_groups[largest_type]) - allocation[largest_type]
        extra = min(remaining, available)
        allocation[largest_type] += extra

    # Sample from each type
    result = []
    for ct, count in allocation.items():
        pool = type_groups[ct]
        if count >= len(pool):
            result.extend(pool)
        else:
            result.extend(rng.sample(pool, count))

    # Shuffle to avoid all type-A then all type-B ordering
    rng.shuffle(result)
    return result


def filter_samples_by_side(samples):
    """
    过滤样本列表，只保留 claim 字段中包含 'left' 或 'right'（不区分大小写）的样本。
    """
    filtered = []
    for sample in samples:
        claim_text = sample.get("claim", "").lower()
        if "left" in claim_text or "right" in claim_text:
            filtered.append(sample)
    return filtered

# ── 无效样本排除清单 ──

NOT_VALID_IDS_JSON = Path(__file__).resolve().parent / "input" / "not_valid_ids.json"


def _load_not_valid_ids(json_path: Path) -> set[str]:
    """加载无效样本 ID 清单（JSON 数组，元素为 {"id": ...} 对象）。

    文件缺失或格式错误时返回空集合并打印警告，不中断主流程。
    """
    if not json_path.exists():
        print(f"Warning: invalid-id file not found, skipping exclusion: {json_path}")
        return set()
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read invalid-id file, skipping exclusion: {json_path} ({exc})")
        return set()
    if not isinstance(data, list):
        print(f"Warning: unexpected format in invalid-id file, skipping exclusion: {json_path}")
        return set()
    ids: set[str] = set()
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.add(str(item["id"]))
        elif isinstance(item, str):
            ids.add(item)
    return ids


# ── 断点续传 (checkpoint) ──

CHECKPOINT_FILE = "checkpoint.json"

# 样本状态:
#   done    已完成(无论答对答错, 进程中断后跳过)
#   running 正在处理(进程中断后视为未完成, 下次重试)
#   fail    处理失败(额度/网络/其它异常, 下次重试)


def _load_checkpoint(checkpoint_path: Path) -> dict:
    """读取上次运行留下的 checkpoint; 文件缺失/损坏时返回空 dict。"""
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Warning: 读取 checkpoint 失败 {checkpoint_path}: {exc}")
        return {}


def _load_previous_results(output_file: Path) -> dict[str, dict]:
    """读取上次运行的结果文件, 按 sample_id 索引 (供断点续传预填已完成样本)。"""
    if not output_file.exists():
        return {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            str(r.get("sample_id")): r
            for r in data.get("samples", [])
            if r.get("sample_id") is not None
        }
    except Exception as exc:
        print(f"Warning: 读取上次结果文件失败 {output_file}: {exc}")
        return {}


def _update_checkpoint(checkpoint_path: Path, state: dict, sample_id, status: str) -> None:
    """线程安全地把某样本状态写入 checkpoint (临时文件 + 原子替换)。"""
    with _results_lock:
        state.setdefault("samples", {})[str(sample_id)] = {"status": status}
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["total_done"] = sum(
            1 for st in state["samples"].values() if st.get("status") == "done"
        )
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            tmp_path.replace(checkpoint_path)
        except Exception as exc:
            print(f"Warning: 写 checkpoint 失败 {checkpoint_path}: {exc}")


def _find_latest_output_dir(base_dir: Path) -> Path | None:
    """在 base_dir 下找"最适合续传"的 outputs_* 目录。

    规则: 含 checkpoint / 结果文件的目录优先, 同一优先级按目录名(时间戳)新的优先。
    返回 None 表示一个都没找到。
    """
    candidates = [p for p in base_dir.glob("outputs_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (
            1 if (p / CHECKPOINT_FILE).exists() or (p / "single_agent_results.json").exists() else 0,
            p.name,
        ),
        reverse=True,
    )
    return candidates[0]


def _build_checkpoint_from_results(checkpoint_path: Path, prev_results: dict[str, dict]) -> dict:
    """旧运行(没有 checkpoint)时, 用结果文件推断样本状态并生成一份 checkpoint。

    推断规则: status 为 error / pipeline_error 的判为 fail(下次重试), 其余判为 done。
    """
    samples = {}
    for sid, r in prev_results.items():
        status = "fail" if r.get("status") in ("error", "pipeline_error") else "done"
        samples[sid] = {"status": status}
    state = {
        "meta": {},
        "samples": samples,
        "inferred_from_results": True,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp_path.replace(checkpoint_path)
    except Exception as exc:
        print(f"Warning: 写 checkpoint 失败 {checkpoint_path}: {exc}")
    return state


# ── Main ──

def main():
    # ── 写入当前进程 PID 到 pid.txt, 方便出问题时 kill ──
    pid = os.getpid()
    pid_file = Path(__file__).resolve().parent / "pid.txt"
    try:
        pid_file.write_text(str(pid), encoding="utf-8")
        print(f"PID {pid} 已写入 {pid_file}")
    except Exception as exc:
        print(f"Warning: 写入 pid.txt 失败: {exc}")

    args = parse_args()

    # ── 输出目录解析 ──
    # --resume 且未显式传 --output-dir 时, 自动定位最近的历史 outputs_* 目录(续传用);
    # 否则用(新生成的)时间戳目录。
    if args.resume and args.output_dir is None:
        latest = _find_latest_output_dir(PROJECT_ROOT)
        if latest is None:
            print("[Resume] 未找到历史 outputs_* 目录, 无法续传, 按全新开始处理")
        else:
            args.output_dir = str(latest)
            print(f"[Resume] 自动定位上次输出目录: {latest}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # [CALL-LOG] 本次运行的 retrieve_unified_knowledge 调用记录写到输出目录
    set_rag_call_log_file(output_dir / "retrieve_unified_calls.jsonl")

    log_dir = output_dir / "llm_raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # [EVIDENCE-FILTER] 证据过滤 LLM 判定日志与各样本 LLM 原始日志同目录
    set_filter_log_file(log_dir / "evidence_filter_llm_log.txt")
    if args.clear_llm_log:
        for existing_log in log_dir.glob("*.txt"):
            existing_log.unlink()
        print(f"Cleared LLM log directory: {log_dir}")

    # 1. Load test data
    # 加载前先读无效 ID 清单：凡 id 在 input/not_valid_ids.json 中的样本一律排除
    not_valid_ids = _load_not_valid_ids(NOT_VALID_IDS_JSON)
    print(f"Loading test data from: {args.test_json}")
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if not_valid_ids:
        original_count = len(test_samples)
        test_samples = [s for s in test_samples if str(s.get("id", "")) not in not_valid_ids]
        excluded_count = original_count - len(test_samples)
        print(
            f"Excluded {excluded_count} invalid sample(s) listed in {NOT_VALID_IDS_JSON} "
            f"({len(test_samples)} remaining)"
        )

    if args.server_data_root:
        print(f"Using server data root override: {args.server_data_root}")
        apply_server_data_root_override(test_samples, args.server_data_root)

    if args.inference_type:
        # 支持单个或多个 inference_type
        if len(args.inference_type) == 1:
            test_samples = [s for s in test_samples if s.get("claim_type") == args.inference_type[0]]
            print(f"Filtered to {args.inference_type[0]} inference: {len(test_samples)} samples")
        else:
            test_samples = [s for s in test_samples if s.get("claim_type") in args.inference_type]
            print(f"Filtered to {', '.join(args.inference_type)} inferences: {len(test_samples)} samples")

    # 按ID文件过滤
    if args.use_id_file:
        id_file_path = Path(args.id_file)
        if id_file_path.exists():
            with open(id_file_path, "r", encoding="utf-8") as f:
                valid_ids = {line.strip() for line in f if line.strip()}
            original_count = len(test_samples)
            test_samples = [s for s in test_samples if str(s.get("id", "")) in valid_ids]
            print(f"Filtered by ID file '{args.id_file}': {len(test_samples)} samples (from {original_count})")
        else:
            print(f"Warning: ID file '{args.id_file}' not found, skipping ID filtering")

    if args.limit:
        test_samples = _stratified_sample_by_type(test_samples, args.limit, seed=args.seed)
        print(f"Sampled {len(test_samples)} samples (stratified by claim_type, seed={args.seed})")



    # 1b. Replace image path prefix
    if args.replace_image_prefix:
        if "=" not in args.replace_image_prefix:
            print("Error: --replace-image-prefix format error, should be old_prefix=new_prefix")
            return
        old_prefix, new_prefix = args.replace_image_prefix.split("=", 1)
        for sample in test_samples:
            if "image_paths" in sample and isinstance(sample["image_paths"], list):
                sample["image_paths"] = [
                    path.replace(old_prefix, new_prefix) if path.startswith(old_prefix) else path
                    for path in sample["image_paths"]
                ]

    # 2. Verify image accessibility
    if not args.skip_image_check:
        print("Verifying image accessibility...")
        accessible_samples = []
        for s in test_samples:
            if all(verify_image_accessible(p) for p in s.get("image_paths", [])):
                accessible_samples.append(s)
        inaccessible_count = len(test_samples) - len(accessible_samples)
        if inaccessible_count > 0:
            print(f"Warning: {inaccessible_count} samples have inaccessible images and will be skipped")
        test_samples = accessible_samples

    # 执行过滤左右侧
    #test_samples = filter_samples_by_side(test_samples)

    print(f"Total samples to process: {len(test_samples)}")

    # 3. Configure execution
    print(f"Using model: {MODEL_NAME}")
    print(f"Agent mode: {args.agent_mode}")
    print(f"Per-sample LLM logs directory: {log_dir}")
    if args.agent_mode == "single_v7":
        print(f"V7 triage source: {args.triage_source}")
        if args.triage_source == "fake":
            seed_fake_rng(args.fake_seed)
            print(f"Fake triage: csv={args.comparison_csv}, seed={args.fake_seed}")

    # 预加载 RAG 编码器(PubMedBERT + MedImageInsight)并常驻:
    # 避免首次工具调用现场加载阻塞数分钟、并拉长 Milvus 连接空闲窗口
    print("Preloading RAG encoders (PubMedBERT + MedImageInsight)...")
    preload_rag_models()

    # 4. Run pipeline on each sample
    output_file = output_dir / "single_agent_results.json"
    stats_file = output_dir / "single_agent_stats.json"

    # ── 断点续传: 跳过已完成样本, 重试上次失败/中断的样本 ──
    checkpoint_path = Path(args.checkpoint_json) if args.checkpoint_json else output_dir / CHECKPOINT_FILE
    progress: dict = {}
    resume_active = False
    if args.resume:
        progress = _load_checkpoint(checkpoint_path)
        state = progress.get("samples", {})
        if not state:
            # 没有 checkpoint / checkpoint 无样本状态 → 从结果文件推断 (支持旧运行的补跑)
            prev_results = _load_previous_results(output_file)
            if prev_results:
                progress = _build_checkpoint_from_results(checkpoint_path, prev_results)
                n_done = sum(1 for s in progress["samples"].values() if s.get("status") == "done")
                n_fail = len(progress["samples"]) - n_done
                print(f"[Resume] 旧运行无 checkpoint, 已按结果文件推断并生成: {checkpoint_path}")
                print(f"[Resume]   成功 {n_done} 条 / 失败待重试 {n_fail} 条")
                resume_active = True
            else:
                print("[Resume] 未找到 checkpoint 也没有历史结果文件, 按全新开始处理")
        else:
            meta = progress.get("meta", {})
            if (meta.get("agent_mode") == args.agent_mode
                    and meta.get("model") == MODEL_NAME
                    and meta.get("test_json") == Path(args.test_json).name):
                resume_active = True
                print(f"[Resume] 读取 checkpoint: {checkpoint_path}")
            else:
                print(f"[Resume] checkpoint 与本次参数不符: {meta}")
                print("[Resume] 可能定位到了错误的目录。请显式指定 --output-dir 指向正确历史目录, 或去掉 --resume 全新开始")
                sys.exit(1)  # 避免自动定位到错误目录后误覆盖历史结果

    results: list = [None] * len(test_samples)
    todo_indices: list[int] = []
    if resume_active:
        done_ids = {
            sid for sid, st in progress.get("samples", {}).items()
            if st.get("status") == "done"
        }
        prev_results = _load_previous_results(output_file)
        for i, s in enumerate(test_samples):
            sid = str(s.get("id", ""))
            if sid in done_ids and sid in prev_results:
                results[i] = prev_results[sid]
        for i in range(len(test_samples)):
            if results[i] is None:
                todo_indices.append(i)
        print(f"[Resume] 上次已完成 {len(done_ids)} 条，本次待处理 {len(todo_indices)} 条")
        if args.clear_llm_log:
            print("[Resume] 注意: --clear-llm-log 会清空已有 LLM 日志，断点续传通常不建议加")
    else:
        todo_indices = list(range(len(test_samples)))

    # checkpoint 元信息 (每次运行都刷新)
    progress.setdefault("meta", {})
    progress["meta"].update({
        "agent_mode": args.agent_mode,
        "model": MODEL_NAME,
        "test_json": Path(args.test_json).name,
        "limit": args.limit,
        "seed": args.seed,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Create DebugTracker for single-agent mode (v2, v3, v4, v5, or v6)
    debug_tracker: DebugTracker | None = None
    if args.agent_mode in ("single", "single_v3", "single_v4", "single_v5", "single_v6", "single_v7", "single_v8", "single_v8_1", "single_v9", "single_v8_corrector"):
        debug_dir = args.debug_dir or str(output_dir / "debug_reports")
        debug_tracker = DebugTracker(debug_dir=debug_dir)
        print(f"Debug reports will be saved to: {debug_dir}")

        # 记录每次 RAG 调用（LLM 请求参数 + 返回的文本/图像）到调试快照,
        # 供 generate_debug_report.py 在报告中紧凑展示。
        from src.tools.multimodal_rag_tool import register_rag_observer
        register_rag_observer(debug_tracker.capture_rag_call)



# todo
    _mode_desc = {
        "multi": "Multi-Agent",
        "single": "Single-Agent-V2",
        "single_v3": "Single-Agent-V3",
        "single_v4": "Single-Agent-V4",
        "single_v5": "Single-Agent-V5",
        "single_v6": "Single-Agent-V6",
        "single_v7": "Single-Agent-V7",
        "single_v8": "Single-Agent-V8",
        "single_v8_1": "Single-Agent-V8_1",
        "single_v8_2": "Single-Agent-V8_2",
        "single_v9": "Single-Agent-V9",
        "single_v8_corrector": "Single-Agent-V8 + Corrector",
    }
    desc = _mode_desc.get(args.agent_mode, "Multi-Agent")
    print(f"Running {desc} pipeline with {args.num_workers} worker(s)...")

    quota_stopped = False

    if args.num_workers <= 1:
        # Fully serial mode (keeps existing behavior as default)
        try:
            for sample_index in tqdm(todo_indices, desc=desc):
                sample = test_samples[sample_index]
                _update_checkpoint(checkpoint_path, progress, sample.get("id", sample_index), "running")
                result = _run_single_sample_with_error_handling(
                    sample,
                    sample_index,
                    log_dir,
                    agent_mode=args.agent_mode,
                    debug_tracker=debug_tracker,
                    triage_source=args.triage_source,
                    comparison_csv=args.comparison_csv,
                )
                results[sample_index] = result
                final_status = (
                    "done" if result.get("status") not in ("error", "pipeline_error") else "fail"
                )
                _update_checkpoint(checkpoint_path, progress, sample.get("id", sample_index), final_status)
                # Incremental write
                current_results = [r for r in results if r is not None]
                _write_results(current_results, output_file)
                _write_stats(current_results, stats_file)
        except QuotaExceededError as qe:
            quota_stopped = True
            print(f"\n{'='*60}")
            print(f"[额度耗尽] 账号 5h 用量额度已耗尽，程序整体停止：{qe}")
            print(f"[停止] 未处理样本保留为 running/fail，下次加 --resume 断点续传即可继续")
            print(f"{'='*60}", flush=True)
    else:
        # Parallel mode (thread-safe DebugTracker supported)
        for sample_index in todo_indices:
            _update_checkpoint(
                checkpoint_path, progress,
                test_samples[sample_index].get("id", sample_index), "running",
            )
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_index = {
                executor.submit(
                    _run_single_sample_with_error_handling,
                    test_samples[sample_index],
                    sample_index,
                    log_dir,
                    args.agent_mode,
                    debug_tracker,
                    args.triage_source,
                    args.comparison_csv,
                ): sample_index
                for sample_index in todo_indices
            }
            try:
                for future in tqdm(as_completed(future_to_index), desc=f"{desc} (Parallel)", total=len(todo_indices)):
                    sample_index = future_to_index[future]
                    result = future.result()
                    results[sample_index] = result
                    final_status = (
                        "done" if result.get("status") not in ("error", "pipeline_error") else "fail"
                    )
                    _update_checkpoint(
                        checkpoint_path, progress,
                        test_samples[sample_index].get("id", sample_index), final_status,
                    )
                    # Incremental write
                    current_results = [r for r in results if r is not None]
                    _write_results(current_results, output_file)
                    _write_stats(current_results, stats_file)
            except QuotaExceededError as qe:
                # 账号额度耗尽 → 取消尚未开始的 futures，整体停止
                quota_stopped = True
                for f in future_to_index:
                    f.cancel()
                print(f"\n{'='*60}")
                print(f"[额度耗尽] 账号 5h 用量额度已耗尽，程序整体停止：{qe}")
                print(f"[停止] 未处理样本保留为 running/fail，下次加 --resume 断点续传即可继续")
                print(f"{'='*60}", flush=True)

    # 5. Final write with stats
    if not quota_stopped:
        _write_final_stats(results, stats_file)

    # 6. Generate debug HTML report (single-agent, serial mode only)
    if debug_tracker is not None:
        jsonl_path = Path(args.debug_dir or "./debug_reports") / "samples.jsonl"
        print(f"\n📊 调试数据已增量保存: {jsonl_path}")
        print(f"   运行以下命令随时生成报告:")
        print(f"     python generate_debug_report.py --data-dir {args.debug_dir or './debug_reports'}")
        print()




def _check_correctness(pred_answer: str, true_label, claim_type: str) -> bool:
    """Check prediction against ground truth (reuses the logic from the existing entry point)."""
    if isinstance(true_label, str) and claim_type == "medical":
        expected = ord(true_label.upper()) - ord('A')
        pred_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        pred = pred_map.get(pred_answer.strip().upper(), -1)
        return pred == expected

    # Yes/No types
    if isinstance(true_label, str):
        return (pred_answer.lower() == "yes") == (true_label == "1")
    if isinstance(true_label, int):
        return (pred_answer.lower() == "yes") == (true_label == 1)

    return False


def _write_results(results: list[dict], output_file: Path) -> None:
    with _results_lock:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"samples": results}, f, indent=2, ensure_ascii=False)


def _is_correct_label(pred: str, label, claim_type: str) -> bool:
    """判断预测答案是否与 label 一致（与 _check_correctness 逻辑相同，接受原始 label 值）。"""
    if isinstance(label, str) and claim_type == "medical":
        expected = ord(label.upper()) - ord('A')
        pred_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        p = pred_map.get(str(pred).strip().upper(), -1)
        return p == expected
    if isinstance(label, str):
        return (str(pred).lower() == "yes") == (label == "1")
    if isinstance(label, int):
        return (str(pred).lower() == "yes") == (label == 1)
    return False


def _compute_corrector_stats(results: list[dict]) -> dict:
    """计算 corrector 闭环的详细统计（整体分类 + 各轮分类）。

    仅对 agent_mode 含 corrector 的样本生效；其他模式下返回空 dict。
    """
    corrector_results = [
        r for r in results
        if r is not None and r.get("agent_mode", "").endswith("corrector")
    ]
    if not corrector_results:
        return {}

    n_total = len(corrector_results)

    # ── 整体分类（基于 initial_answer vs final_answer vs true_label） ──
    # 类别：
    #   unchanged_ok    初始对 → 最终对（corrector 没改 / 改了还是对）
    #   fixed           初始错 → 最终对（修复成功）
    #   broke           初始对 → 最终错（改错了，引入新错误）
    #   unchanged_err   初始错 → 最终错（没修好）
    overall = {"unchanged_ok": 0, "fixed": 0, "broke": 0, "unchanged_err": 0}
    baseline_correct = 0
    final_correct = 0
    # 失败样本拆分：status="error" 的样本 initial/final 都是 "Uncertain"，
    # 会被计入分母并算错题 → 拉低 baseline/final accuracy。
    # 下面单独统计成功样本口径，两种口径同时落盘。
    n_failed = 0
    success_baseline_correct = 0
    success_final_correct = 0

    # ── 各轮详细分类 ──
    # per_round[r] = 经过第 r 轮审查+重答后的统计
    #   reviewed: corrector 审查了（判为 error）
    #   revised_and_changed: 重答后答案变了
    #   revised_unchanged: 重答后答案没变
    #   ok_at_review: corrector 判为 ok（到此为止）
    per_round: dict[int, dict] = {}
    max_round_seen = 0

    # ── corrector 行为统计 ──
    corrector_verdicts = {
        "flagged_error": 0,         # corrector 判"有错"的总次数
        "passed_ok": 0,             # corrector 判"无错"的总次数
        "false_positive": 0,        # corrector 判错且初始答案其实是对的（乱挑错）
        "true_positive": 0,         # corrector 判错且初始答案确实是错的
        "false_negative": 0,        # corrector 判无错但初始答案其实是错的（漏放）
        "true_negative": 0,         # corrector 判无错且初始答案确实是对的
    }

    # ── 仲裁统计（ARBITER_ENABLED=True 时生效） ──
    arbiter_stats = {
        "invoked": 0,               # 仲裁触发次数（答案变化时才触发）
        "kept_old": 0,              # judge 判保留旧解（A / tie / 失败兜底）
        "kept_new": 0,              # judge 判采纳新解（B）
        "kept_old_blocked_broke": 0,  # 拦截正确：旧对→新错，judge 保旧（避免 broke）
        "kept_old_blocked_fix": 0,    # 误拦截：旧错→新对，judge 保旧（错失修复）
        "kept_old_neutral": 0,        # 保旧但新旧对错状态相同（无实际影响）
        "kept_new_fixed": 0,          # 放行正确：旧错→新对（真修复被采纳）
        "kept_new_broke": 0,          # 放行失败：旧对→新错（broke 没拦住）
        "kept_new_neutral": 0,        # 放行但新旧对错状态相同
        "failed": 0,                # 调用/解析失败次数（按 tie 保旧）
    }

    for r in corrector_results:
        label = r.get("true_label")
        claim_type = r.get("claim_type", "unknown")
        init_ans = r.get("initial_answer", "Uncertain")
        final_ans = r.get("final_answer", "Uncertain")
        init_ok = _is_correct_label(init_ans, label, claim_type)
        final_ok = _is_correct_label(final_ans, label, claim_type)

        if init_ok and final_ok:
            overall["unchanged_ok"] += 1
        elif not init_ok and final_ok:
            overall["fixed"] += 1
        elif init_ok and not final_ok:
            overall["broke"] += 1
        else:
            overall["unchanged_err"] += 1

        if init_ok:
            baseline_correct += 1
        if final_ok:
            final_correct += 1

        # 成功/失败拆分（失败样本两口径都算错，但分母分开算）
        if r.get("status") == "success":
            if init_ok:
                success_baseline_correct += 1
            if final_ok:
                success_final_correct += 1
        else:
            n_failed += 1

        # 各轮统计（从 round_history 解析）
        history = r.get("round_history") or []
        rounds_used = r.get("correction_rounds", 0)
        if rounds_used > max_round_seen:
            max_round_seen = rounds_used

        # 第 1 轮审查的 verdict 用来算 corrector 的 TP/FP/TN/FN
        # （只看第 1 轮，因为后续轮次答案已经变了）
        for entry in history:
            if entry.get("round") == 1 and entry.get("reviewer_verdict") is not None:
                verdict = entry["reviewer_verdict"]
                if verdict == "error":
                    corrector_verdicts["flagged_error"] += 1
                    if init_ok:
                        corrector_verdicts["false_positive"] += 1
                    else:
                        corrector_verdicts["true_positive"] += 1
                elif verdict == "ok":
                    corrector_verdicts["passed_ok"] += 1
                    if init_ok:
                        corrector_verdicts["true_negative"] += 1
                    else:
                        corrector_verdicts["false_negative"] += 1
                break

        # 按轮号统计每轮的行为
        for entry in history:
            rnd = entry.get("round", 0)
            if rnd == 0:
                continue
            phase = entry.get("phase", "")
            if rnd not in per_round:
                per_round[rnd] = {
                    "reviewed_error": 0,     # corrector 判 error，触发重答
                    "reviewed_ok": 0,        # corrector 判 ok，停止
                    "answer_changed": 0,     # 重答后答案变了
                    "answer_unchanged": 0,   # 重答后答案没变
                    "fixed_in_round": 0,     # 本轮后答案变对了（从错到对）
                    "broke_in_round": 0,     # 本轮后答案变错了（从对到错）
                }
            pr = per_round[rnd]
            if phase == "review":
                pr["reviewed_ok"] += 1
            elif phase == "review_then_revise":
                pr["reviewed_error"] += 1
                # revise 本身是否改了答案（不受仲裁回退影响 → 固执率口径）
                revised_changed = entry.get(
                    "revised_answer_changed", entry.get("answer_changed", False)
                )
                if revised_changed:
                    pr["answer_changed"] += 1
                else:
                    pr["answer_unchanged"] += 1
                # 判断本轮是修复了还是搞砸了（用仲裁后实际采用的答案）
                ans_before = entry.get("answer_before", "Uncertain")
                ans_final = entry.get("answer_final", entry.get("answer_after", "Uncertain"))
                before_ok = _is_correct_label(ans_before, label, claim_type)
                final_ok = _is_correct_label(ans_final, label, claim_type)
                if not before_ok and final_ok:
                    pr["fixed_in_round"] += 1
                elif before_ok and not final_ok:
                    pr["broke_in_round"] += 1

                # ── 仲裁反事实统计 ──
                # entry["answer_after"] 是 revise 的原始新答案（仲裁前的），

                if entry.get("arbiter_applied") in ("keep_old", "keep_new"):
                    arbiter_stats["invoked"] += 1
                    ans_rev = entry.get("answer_after", "Uncertain")
                    rev_ok = _is_correct_label(ans_rev, label, claim_type)
                    if entry.get("arbiter_status") == "failed":
                        arbiter_stats["failed"] += 1
                    if entry["arbiter_applied"] == "keep_old":
                        arbiter_stats["kept_old"] += 1
                        if before_ok and not rev_ok:
                            arbiter_stats["kept_old_blocked_broke"] += 1
                        elif not before_ok and rev_ok:
                            arbiter_stats["kept_old_blocked_fix"] += 1
                        else:
                            arbiter_stats["kept_old_neutral"] += 1
                    else:  # keep_new
                        arbiter_stats["kept_new"] += 1
                        if not before_ok and rev_ok:
                            arbiter_stats["kept_new_fixed"] += 1
                        elif before_ok and not rev_ok:
                            arbiter_stats["kept_new_broke"] += 1
                        else:
                            arbiter_stats["kept_new_neutral"] += 1

    # 计算比率
    baseline_acc = baseline_correct / n_total if n_total else 0.0
    final_acc = final_correct / n_total if n_total else 0.0

    # corrector 检测指标（以 error 为正类）
    tp = corrector_verdicts["true_positive"]
    fp = corrector_verdicts["false_positive"]
    tn = corrector_verdicts["true_negative"]
    fn = corrector_verdicts["false_negative"]
    detection_acc = (tp + tn) / max(1, tp + fp + tn + fn)
    detection_prec = tp / max(1, tp + fp)
    detection_rec = tp / max(1, tp + fn)
    detection_f1 = 2 * detection_prec * detection_rec / max(1e-9, detection_prec + detection_rec)

    # 对错题 / 对对题挑错指标（显式落盘，便于跨实验对比 corrector 区分度）
    #   对错题挑错 = TP（初始答错且被挑出），率 = TP / 初始答错总数（= recall）
    #   对对题挑错 = FP（初始答对却被挑错），率 = FP / 初始答对总数（误报率）
    initial_wrong_total = tp + fn       # 初始答错总数
    initial_correct_total = fp + tn     # 初始答对总数
    flagged_on_wrong = tp               # 对错题挑错数
    flagged_on_correct = fp             # 对对题挑错数
    flagged_on_wrong_rate = tp / max(1, initial_wrong_total)      # 对错题挑错率
    flagged_on_correct_rate = fp / max(1, initial_correct_total)  # 对对题挑错率

    # 成功样本口径的准确率（失败样本 status="error" 的 initial/final 均
    # 为 "Uncertain"，计入总分母会拉低 accuracy；成功口径剔除这部分噪声）
    n_success = n_total - n_failed
    baseline_acc_success = success_baseline_correct / n_success if n_success else 0.0
    final_acc_success = success_final_correct / n_success if n_success else 0.0

    return {
        "total_samples": n_total,
        "sample_breakdown": {
            "success": n_success,
            "failed": n_failed,  # status="error"，initial/final 均为 Uncertain，计为错题
        },
        "baseline_accuracy": baseline_acc,
        "final_accuracy": final_acc,
        "accuracy_delta": final_acc - baseline_acc,
        "baseline_accuracy_success_only": baseline_acc_success,
        "final_accuracy_success_only": final_acc_success,
        "accuracy_delta_success_only": final_acc_success - baseline_acc_success,
        "overall_classification": overall,
        "overall_rates": {
            "fix_rate": overall["fixed"] / max(1, overall["fixed"] + overall["unchanged_err"]),
            "break_rate": overall["broke"] / max(1, overall["unchanged_ok"] + overall["broke"]),
            "revision_stubborn_rate": 0.0,  # 下面单独算
        },
        "corrector_detection": {
            "accuracy": detection_acc,
            "precision": detection_prec,
            "recall": detection_rec,
            "f1": detection_f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "flagged_error_total": corrector_verdicts["flagged_error"],
            "passed_ok_total": corrector_verdicts["passed_ok"],
            # 对错题 / 对对题挑错指标（含分母，可直接读出区分度）
            "initial_wrong_total": initial_wrong_total,
            "initial_correct_total": initial_correct_total,
            "flagged_on_wrong": flagged_on_wrong,                    # 对错题挑错数
            "flagged_on_wrong_rate": flagged_on_wrong_rate,          # 对错题挑错率
            "flagged_on_correct": flagged_on_correct,                # 对对题挑错数
            "flagged_on_correct_rate": flagged_on_correct_rate,      # 对对题挑错率
        },
        "per_round_stats": {
            str(r): per_round[r] for r in sorted(per_round.keys())
        },
        "max_rounds_used": max_round_seen,
        # 仲裁统计：始终落盘（即使 ARBITER_ENABLED=False 或未触发，
        # 也要写明状态和配置，否则 stats 里完全看不到仲裁相关的痕迹）
        "arbiter": {
            "enabled": bool(ARBITER_ENABLED),
            "config": {"max_retries": ARBITER_MAX_RETRIES},
            "stats": arbiter_stats,
        },
    }


def _write_stats(results: list[dict], stats_file: Path) -> None:
    results = [r for r in results if r is not None]  # 额度停止等场景下可能含 None
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total > 0 else 0.0

    # Per-type stats
    type_stats = {}
    for r in results:
        ct = r.get("claim_type", "unknown")
        if ct not in type_stats:
            type_stats[ct] = {"total": 0, "correct": 0}
        type_stats[ct]["total"] += 1
        if r.get("correct"):
            type_stats[ct]["correct"] += 1

    for ct, st in type_stats.items():
        st["accuracy"] = st["correct"] / st["total"] if st["total"] > 0 else 0.0

    # ── Initial (pre-reflection) accuracy ──
    # 记录自我反思 / 格式纠正 之前的初始答案准确率
    has_initial = any("initial_answer" in r for r in results)
    initial_stats: dict = {}
    if has_initial:
        initial_correct = 0
        initial_valid = 0
        initial_type_stats = {}
        for r in results:
            init_ans = r.get("initial_answer")
            if init_ans is None:
                continue
            initial_valid += 1
            ct = r.get("claim_type", "unknown")
            label = r.get("true_label")
            init_correct = _is_correct_label(init_ans, label, ct)
            if init_correct:
                initial_correct += 1
            if ct not in initial_type_stats:
                initial_type_stats[ct] = {"total": 0, "correct": 0}
            initial_type_stats[ct]["total"] += 1
            if init_correct:
                initial_type_stats[ct]["correct"] += 1
        for ct, st in initial_type_stats.items():
            st["accuracy"] = st["correct"] / st["total"] if st["total"] > 0 else 0.0

        # 答案变化统计
        changed_count = sum(
            1 for r in results
            if r.get("initial_answer") is not None
            and r.get("initial_answer") != r.get("final_answer")
        )
        changed_to_correct = sum(
            1 for r in results
            if r.get("initial_answer") is not None
            and r.get("initial_answer") != r.get("final_answer")
            and r.get("correct")
        )
        changed_to_wrong = sum(
            1 for r in results
            if r.get("initial_answer") is not None
            and r.get("initial_answer") != r.get("final_answer")
            and not r.get("correct")
            and _is_correct_label(r["initial_answer"], r.get("true_label"), r.get("claim_type", "unknown"))
        )

        initial_stats = {
            "total_samples": initial_valid,
            "correct_count": initial_correct,
            "accuracy": initial_correct / initial_valid if initial_valid > 0 else 0.0,
            "per_type_stats": initial_type_stats,
            "answer_changes": {
                "total_changed": changed_count,
                "changed_to_correct": changed_to_correct,
                "changed_to_wrong": changed_to_wrong,
            },
        }

    # corrector 专项统计
    corrector_stats = _compute_corrector_stats(results)

    output = {
        "total_samples": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "per_type_stats": type_stats,
    }
    if initial_stats:
        output["initial"] = initial_stats
    if corrector_stats:
        output["corrector"] = corrector_stats

    with _results_lock:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


def _print_corrector_report(stats: dict) -> None:
    """把 corrector 统计打印成人类可读的表格。"""
    if not stats:
        return

    oc = stats["overall_classification"]
    n = stats["total_samples"]
    det = stats["corrector_detection"]

    print(f"\n{'='*80}")
    print(f"📊 Corrector 闭环分析（n = {n}）")
    print(f"{'='*80}")

    print(f"\n── 整体效果 ──")
    print(f"  基线准确率    : {stats['baseline_accuracy']:.4f}")
    print(f"  闭环后准确率  : {stats['final_accuracy']:.4f}")
    print(f"  提升          : {stats['accuracy_delta']:+.4f}")
    sb = stats.get("sample_breakdown", {})
    if sb.get("failed", 0):
        print(f"  ⚠ 失败样本    : {sb['failed']}/{stats['total_samples']}（status=error，计为错题拉低了两项准确率）")
        print(f"  成功样本口径  : 基线 {stats['baseline_accuracy_success_only']:.4f} → 闭环 {stats['final_accuracy_success_only']:.4f}"
              f"（{stats.get('accuracy_delta_success_only', 0):+.4f}，n={sb['success']}）")

    print(f"\n── 整体分类 ──")
    print(f"  保持正确 (unchanged_ok)  : {oc['unchanged_ok']:>4d}  ({oc['unchanged_ok']/n*100:5.1f}%)")
    print(f"  修复成功 (fixed)         : {oc['fixed']:>4d}  ({oc['fixed']/n*100:5.1f}%)  ← 错→对")
    print(f"  引入错误 (broke)         : {oc['broke']:>4d}  ({oc['broke']/n*100:5.1f}%)  ← 对→错")
    print(f"  仍未修复 (unchanged_err) : {oc['unchanged_err']:>4d}  ({oc['unchanged_err']/n*100:5.1f}%)")

    print(f"\n── Corrector 检测能力（第 1 轮，error=正类） ──")
    print(f"  检测准确率 : {det['accuracy']:.4f}")
    print(f"  精确率     : {det['precision']:.4f}  (判错里真的错的比例)")
    print(f"  召回率     : {det['recall']:.4f}  (真的错里被挑出来的比例)")
    print(f"  F1         : {det['f1']:.4f}")
    print(f"  混淆矩阵   : TP={det['tp']}  FP={det['fp']}  FN={det['fn']}  TN={det['tn']}")
    print(f"  对错题挑错 : {det['flagged_on_wrong']}/{det['initial_wrong_total']}"
          f"  ({det['flagged_on_wrong_rate']*100:.1f}%)  ← 期望高")
    print(f"  对对题挑错 : {det['flagged_on_correct']}/{det['initial_correct_total']}"
          f"  ({det['flagged_on_correct_rate']*100:.1f}%)  ← 期望低")

    # 各轮统计
    per_round = stats.get("per_round_stats", {})
    if per_round:
        print(f"\n── 各轮行为统计 ──")
        for rnd_key in sorted(per_round.keys(), key=lambda k: int(k)):
            pr = per_round[rnd_key]
            total_review = pr["reviewed_error"] + pr["reviewed_ok"]
            changed_rate = pr["answer_changed"] / max(1, pr["reviewed_error"])
            print(f"  第 {rnd_key} 轮:")
            print(f"    审查总数   : {total_review}")
            print(f"    判 error   : {pr['reviewed_error']}  → 触发重答")
            print(f"    判 ok      : {pr['reviewed_ok']}  → 停止")
            if pr["reviewed_error"] > 0:
                print(f"    重答后改变 : {pr['answer_changed']}  ({changed_rate*100:.1f}%)")
                print(f"    重答后不变 : {pr['answer_unchanged']}  ({(1-changed_rate)*100:.1f}%)  ← 主模型\"不改\"比例")
                print(f"    本轮修复   : {pr['fixed_in_round']}")
                print(f"    本轮引入   : {pr['broke_in_round']}")

    # 主模型"固执"程度：被挑错后答案不变的比例
    r1 = per_round.get("1", {})
    if r1 and r1.get("reviewed_error", 0) > 0:
        stubborn_rate = r1["answer_unchanged"] / r1["reviewed_error"]
        print(f"\n── 主模型\"固执\"指数（第 1 轮） ──")
        print(f"  corrector 挑错后主模型不改答案的比例: {stubborn_rate*100:.1f}%")
        if stubborn_rate > 0.5:
            print(f"  ⚠️  超过 50%，主模型可能倾向于不接受反馈")

    # 仲裁统计（始终输出；未启用/未触发时也写明状态）
    arb_block = stats.get("arbiter")
    if arb_block:
        print(f"\n── 仲裁（Arbiter）统计 ──")
        print(f"  开关状态 : {'启用' if arb_block['enabled'] else '未启用（ARBITER_ENABLED=False）'}")
        arb = arb_block.get("stats") or {}
        print(f"  触发次数 : {arb.get('invoked', 0)}  （仅重答后答案变化时触发）")
        if arb.get("invoked", 0) > 0:
            print(f"  保留旧解 : {arb['kept_old']}  其中拦截了 {arb['kept_old_blocked_broke']} 个潜在 broke"
                  f"、误拦 {arb['kept_old_blocked_fix']} 个潜在修复、无影响 {arb['kept_old_neutral']}")
            print(f"  采纳新解 : {arb['kept_new']}  其中放行 {arb['kept_new_fixed']} 个真修复"
                  f"、漏拦 {arb['kept_new_broke']} 个 broke、无影响 {arb['kept_new_neutral']}")
            if arb.get("failed"):
                print(f"  调用失败 : {arb['failed']}  （按 tie 保旧处理）")
            net = arb["kept_old_blocked_broke"] + arb["kept_new_fixed"] \
                - arb["kept_old_blocked_fix"] - arb["kept_new_broke"]
            print(f"  仲裁净贡献（相对无仲裁）: {net:+d} 题")

    print(f"\n{'='*80}")


def _write_final_stats(results: list[dict], stats_file: Path) -> None:
    _write_stats(results, stats_file)

    results = [r for r in results if r is not None]
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total > 0 else 0.0

    # Per-type stats
    type_stats = {}
    for r in results:
        ct = r.get("claim_type", "unknown")
        if ct not in type_stats:
            type_stats[ct] = {"total": 0, "correct": 0}
        type_stats[ct]["total"] += 1
        if r.get("correct"):
            type_stats[ct]["correct"] += 1

    for ct, st in type_stats.items():
        st["accuracy"] = st["correct"] / st["total"] if st["total"] > 0 else 0.0

    # Initial (pre-reflection) accuracy
    has_initial = any("initial_answer" in r for r in results)

    print(f"\n{'='*80}")
    print(f"Results saved to: {stats_file}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    for ct, st in type_stats.items():
        print(f"  {ct}: {st['accuracy']:.4f} ({st['correct']}/{st['total']})")

    if has_initial:
        init_correct = sum(
            1 for r in results
            if r.get("initial_answer") is not None
            and _is_correct_label(r["initial_answer"], r.get("true_label"), r.get("claim_type", "unknown"))
        )
        init_total = sum(1 for r in results if r.get("initial_answer") is not None)
        init_acc = init_correct / init_total if init_total > 0 else 0.0
        changed = sum(
            1 for r in results
            if r.get("initial_answer") is not None
            and r.get("initial_answer") != r.get("final_answer")
        )
        print(f"\nInitial Accuracy (pre-reflection): {init_acc:.4f} ({init_correct}/{init_total})")
        print(f"Final Accuracy   (post-reflection): {accuracy:.4f} ({correct}/{total})")
        if init_total > 0:
            delta = accuracy - init_acc
            print(f"Delta: {delta:+.4f}")
        print(f"Answers changed after reflection: {changed}/{init_total}")

    # corrector 专项报告
    corrector_stats = _compute_corrector_stats(results)
    if corrector_stats:
        _print_corrector_report(corrector_stats)

    print(f"{'='*80}")


if __name__ == "__main__":
    main()

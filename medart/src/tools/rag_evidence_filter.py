"""
RAG 证据过滤模块 — 实验性, 独立可整体移除。

在 RAG 工具把检索结果返回给大模型之前, 逐条判断「该命中的文本是否构成对
查询的支撑性证据 (supporting evidence)」, 不是证据的命中直接从返回列表中
剔除, 避免大模型被无关检索碎片带偏。

支持两种后端, 通过环境变量 RAG_EVIDENCE_FILTER_BACKEND 切换:
- llm   (默认): 商业大模型, 走火山方舟 / 任意 OpenAI 兼容 API, 直接调用
  大模型做 Yes/No 判定, 不占本地显存、无硬件限制。
- local: 本地 Llama-3.1-8B-Evidence-Filtering (Yale-BIDS-Chen), 4-bit 量化
  加载。注意: 该机 P100 无 int8 Tensor Core, 8-bit 的 int8_mixed_scaled_mm
  必崩 (cublasLt status 15), 本地后端仅支持 4-bit NF4 + FP16 计算。

设计目标: 尽量少侵入原有 rag tool 代码, 方便后期整体去掉。
- 全部过滤逻辑收敛在本文件; 各 rag tool 只需在得到 fused 融合结果后调用钩子
      fused = filter_rag_hits_by_evidence(query, fused, text_key="text")
  (需 import:
      from src.tools.rag_evidence_filter import filter_rag_hits_by_evidence)
- 移除方式: 删除各 rag tool 里的钩子行 + 本文件即可, 不影响其余任何逻辑。
- 兜底 (fail-open): 后端不可用 / 单条调用失败 / 被开关禁用时, 一律原样返回
  原有命中列表(不过滤), 保证检索管线照常工作 —— 过滤只是"锦上添花"。
- 开关:
  - RAG_EVIDENCE_FILTER=0 (或 false/off/no/disabled) 整体禁用; 默认启用。
  - RAG_EVIDENCE_FILTER_BACKEND=llm|local 选择后端; 默认 llm。
  - LLM 后端的 key/模型/base_url 可用环境变量覆盖
    (RAG_EVIDENCE_LLM_API_KEY / RAG_EVIDENCE_LLM_MODEL / RAG_EVIDENCE_LLM_BASE_URL),
    默认值与 run_multi_agent._build_llm 一致。

并发: 本地后端的推理通过本模块自带的 _INFER_LOCK 互斥; 同时复用
multimodal_rag_tool 的 _RAG_GATE(懒加载 import, 避免循环依赖), 与编码/检索
共用同一把全局门闸, 防止与并发 RAG 调用抢 GPU 显存 (OOM)。
注意: 因此钩子必须放在各 rag tool 的 `with _RAG_GATE:` 块【之后】调用
(工具本身不再持有门闸), 否则 FIFO 信号量不可重入会死锁。
LLM 后端不占显存, 不打门闸, 只做轻量互斥。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("rag_evidence_filter")

# ---------------------------------------------------------------------------
# 开关 + 后端选择
# ---------------------------------------------------------------------------
RAG_EVIDENCE_FILTER =0

def _filter_enabled() -> bool:
    return bool(RAG_EVIDENCE_FILTER)


def _backend() -> str:
    #v = os.environ.get("RAG_EVIDENCE_FILTER_BACKEND", "llm").strip().lower()
    #return "local" if v in ("local", "llama", "4bit") else "llm"
    return "llm"


# ---------------------------------------------------------------------------
# 当前样本 claim(题目)注入 —— 判定用 claim, 而不是 LLM 生成的检索关键词
# ---------------------------------------------------------------------------

# 用 contextvar 而不是全局变量: 多线程各司其职互不干扰 (agent 层每次样本
# run 时用 set_claims_for_filter() 设置, 工具层读取)。
_ACTIVE_CLAIM: ContextVar[str] = ContextVar("rag_evidence_filter_claim", default="")


def set_claims_for_filter(claim: str) -> None:
    """[EVIDENCE-FILTER] 设置当前样本的题目文本, 用于证据过滤判定。

    让判断依据从「检索关键词 (query)」改为「题目 (claim)」: 检索到的内容
    是否构成对题目的支撑性证据, 而不是是否命中搜索词。传给 "" 则退回用
    query 判定 (fail-open)。
    """
    _ACTIVE_CLAIM.set(claim or "")


def _active_claim() -> str:
    return _ACTIVE_CLAIM.get().strip()


# ---------------------------------------------------------------------------
# 后端一: 商业大模型 (OpenAI 兼容 API, 不依赖 RawChatOpenAI)
# ---------------------------------------------------------------------------

# 默认值与 run_multi_agent._build_llm 一致; 均可用环境变量覆盖
LLM_API_KEY = "*********"
LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
LLM_MODEL = "doubao-seed-2.1-turbo"

_LLM_CLIENT: Optional[Any] = None
_LLM_CLIENT_LOCK = threading.Lock()

# LLM 后端的调用日志(输入/输出), 仿 RawChatOpenAI 的 ==== 段落式 JSON
# 格式; 默认写在项目根目录, 可用环境变量 RAG_EVIDENCE_FILTER_LOG 覆盖;
# 也可以在 run_multi_agent 里用 set_filter_log_file() 指到输出目录。
FILTER_LOG_FILE = os.environ.get(
    "RAG_EVIDENCE_FILTER_LOG",
    str(Path(__file__).resolve().parents[2] / "evidence_filter_llm_log.txt"),
)


def set_filter_log_file(path: str | os.PathLike) -> None:
    """[EVIDENCE-FILTER] 重定向证据过滤 LLM 日志文件。

    供 run_multi_agent 把日志写到本次运行的输出目录 (llm_raw_logs) 下,
    与各样本的 LLM 原始日志放一起; 也可以在命令行用环境变量覆盖。
    """
    global FILTER_LOG_FILE
    FILTER_LOG_FILE = str(path)


def _write_filter_log(label: str, payload: dict[str, Any]) -> None:
    """把一条 LLM 调用(输入或输出)以段落式 JSON 追加到日志文件。"""
    try:
        log_text = f"\n{'='*20} {datetime.now()} {label} {'='*20}\n"
        log_text += json.dumps(payload, ensure_ascii=False, indent=2)
        log_text += f"\n{'='*80}\n"
        with open(FILTER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_text)
    except Exception:
        logger.exception("Failed to write evidence filter LLM log.")


def _get_llm_client() -> Any:
    """懒加载 OpenAI 兼容客户端(只做配置初始化, 不发网络请求)。"""
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    with _LLM_CLIENT_LOCK:
        if _LLM_CLIENT is None:
            from openai import OpenAI

            _LLM_CLIENT = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            logger.info(
                "Evidence filter LLM backend ready: model=%s base=%s",
                LLM_MODEL, LLM_BASE_URL,
            )
    return _LLM_CLIENT


# 训练时的判定 prompt (与 tests/filter_test.py 的 INSTRUCTION 一致,
# 商业模型同样用这套指令)
#
# 判定标准说明: 不要要求 passage "直接给出答案"。RAG 召回的是背景医学知识
# (疾病的特征 / 鉴别 / 病因 / 处理原则等), 用来辅助推理, 几乎不可能逐字
# 对应题目或提到具体患者的影像。原版要求 "directly backs or justifies the
# answer" 过严, 导致大量相关命中被判 No 删除 (removed 5/5)。
INSTRUCTION = (
    "Given a medical question and a text passage, determine whether the "
    "passage is relevant background material that could help answer the "
    "question.\n\n"
    "The passage counts as supporting evidence if it is topically related to "
    "the question: for example, it discusses the same condition, its clinical "
    "features, risk factors, etiology, pathophysiology, diagnosis, "
    "differential diagnosis, complications, or management. It does NOT need to "
    "mention the specific patient, the image, or state the final answer — "
    "background knowledge that helps reason toward the answer is enough.\n\n"
    'Answer "No" only if the passage is clearly off-topic: a different '
    "disease, a different body system, or content with no bearing on the "
    "question.\n\n"
    'Respond with one of the following labels:\n"Yes" if the passage is '
    'relevant background for the question.\n"No" if the passage is clearly '
    "unrelated to the question.\n"
    "You should respond with only the label (Yes or No) without any additional "
    "explanation."
)


def _infer_labels_llm(query: str,
                      passages: list[Optional[str]],
                      origin: str = "query",
                      rag_query: str = "",
                      rag_sources: Optional[list[Optional[list[str]]]] = None,
                      collection: str = "",
                      ) -> list[Optional[str]]:
    """逐条调用商业大模型判定, 返回与 passages 等长的标签列表。

    单条失败记 warning 并记 None(该条在过滤阶段按 fail-open 保留),
    不中断整批。空 passage 记 None。

    日志: 每条判定写成【一个完整段落】—— rag_query (检索关键词) /
    recall_sources (该命中来自文本路还是图像路) / query (判定问题, 一般
    是 claim) / passage / label / reasoning 全在一起, 人一眼就能对上传入
    的检索词、命中来源、被判对象、判定结果 (而不是分开的 REQUEST /
    RESPONSE 两段)。注意双路检索的图像路召回与关键词无关, 记来源可避免
    「关键词与 passage 不相关」的误解。
    """
    client = _get_llm_client()
    labels: list[Optional[str]] = []
    for i, pas in enumerate(passages):
        pas = (pas or "").strip()
        if not pas:
            labels.append(None)
            continue
        source = (rag_sources[i] if rag_sources and i < len(rag_sources) else None)
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user",
                     "content": f"Question: {query}\nPassage: {pas}"},
                ],
                temperature=0.1,
                # 调大: doubao-seed 这类 reasoning 模型 16 token 不够, content
                # 会被思维链吃空 → 误判为删; 64 token 足够输出 Yes/No
                max_tokens=64,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 同时记 reasoning content(若有), 便于排查思维链吃空输出的情况
            reasoning = ""
            if hasattr(resp.choices[0].message, "reasoning_content"):
                reasoning = (resp.choices[0].message.reasoning_content or "").strip()
            # 合并写一条完整记录: 检索词 + 命中来源 + collection + 判定问题 +
            # passage + 判定
            _write_filter_log("EVIDENCE-FILTER LLM JUDGMENT", {
                "rag_query": rag_query,
                "recall_sources": source,
                "collection": collection,
                "query_origin": origin,
                "query": query,
                "passage_index": i,
                "passage": pas,
                "label": text,
                "reasoning_content": reasoning,
            })
            labels.append(text)
        except Exception as e:
            logger.warning(
                "Evidence filter LLM call failed (%s); keep this hit (fail-open).",
                e,
            )
            _write_filter_log("EVIDENCE-FILTER LLM JUDGMENT", {
                "rag_query": rag_query,
                "recall_sources": source,
                "collection": collection,
                "query_origin": origin,
                "query": query,
                "passage_index": i,
                "passage": pas,
                "label": None,
                "error": str(e),
            })
            labels.append(None)
    return labels


# ---------------------------------------------------------------------------
# 后端二: 本地 Llama-3.1-8B-Evidence-Filtering (懒加载单例, 4-bit)
# ---------------------------------------------------------------------------

_ENGINE: Optional[dict[str, Any]] = None
_ENGINE_FAILED: bool = False
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()  # 串行化本地推理, 与 _RAG_GATE 一起防 OOM


def _resolve_model_dir() -> str:
    # 与 tests/filter_test.py 一致
    return "/home/jzs/od/model/Llama-3.1-8B-Evidence-Filtering"


def _get_engine() -> Optional[dict[str, Any]]:
    """懒加载本地过滤模型。失败则缓存失败状态, 之后直接返回 None。"""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    with _LOAD_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _ENGINE_FAILED:
            return None
        model_dir = _resolve_model_dir()
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            # P100 (Pascal, CC 6.0) 硬件限制: 无 int8 Tensor Core → 8-bit 走
            # int8_mixed_scaled_mm 必崩 (cublasLt status 15); 也无 bf16 硬件。
            # 因此用 NF4 4-bit 存储 + FP16 计算, 是 P100 16G 上能跑的最高精度。
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # decoder-only 批量生成必须左填充, 否则 generate 结果错
            # (transformers 会警告 right-padding detected)
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model.eval()
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            _ENGINE = {
                "model": model,
                "tokenizer": tokenizer,
                "terminators": terminators,
                "torch": torch,
            }
            logger.info("Evidence filter local model loaded: %s", model_dir)
        except Exception:
            logger.exception(
                "Evidence filter local model load failed; filter disabled (fail-open)."
            )
            _ENGINE_FAILED = True
            return None
    return _ENGINE


def _infer_labels_local(query: str, passages: list[Optional[str]],
                        engine: dict[str, Any],
                        origin: str = "query",
                        rag_sources: Optional[list[Optional[list[str]]]] = None,
                        ) -> list[Optional[str]]:
    """批量构造 chat prompt 并做一次 generate, 返回与 passages 等长的标签列表。

    每行 decode 只取「新生成的 token」部分, 因此各 passage 输入长度不一致
    也能逐行正确提取输出。空 passage 直接记 None(不计入待推理批)。
    """
    tokenizer = engine["tokenizer"]
    contents: list[Optional[str]] = []
    idxs: list[int] = []
    for i, pas in enumerate(passages):
        pas = (pas or "").strip()
        if not pas:
            contents.append(None)
            continue
        contents.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": f"Question: {query}\nPassage: {pas}"},
            ],
            add_generation_prompt=True,
            tokenize=False,
        ))
        idxs.append(i)

    labels: list[Optional[str]] = [None] * len(passages)
    if not idxs:
        return labels

    batch = [contents[i] for i in idxs]
    model = engine["model"]
    torch = engine["torch"]
    enc = tokenizer(batch, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_lens = [len(ids) for ids in enc["input_ids"]]  # 每行原始长度

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=16,
            eos_token_id=engine["terminators"],
            do_sample=False,
            temperature=0.0,
        )

    for out_i, orig_i in enumerate(idxs):
        new_tokens = outputs[out_i][input_lens[out_i]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        labels[orig_i] = text
    return labels


# ---------------------------------------------------------------------------
# 判定 + 对外钩子
# ---------------------------------------------------------------------------

def _is_no(label: Optional[str]) -> bool:
    """三态判定: 只有明确以 No 开头的标签才删除, 其余(空/乱答/Yes)一律保留。

    之前是 _is_yes: 只要不以 Yes 开头就当 No 删 → 模型空返回/格式乱时
    把全部命中都误删 (日志里 removed 5/5)。这违反 fail-open 精神。
    """
    if not label:
        return False
    return label.lower().startswith("no")


def _evidence_question(query: str, claim: str) -> tuple[str, str]:
    """确定判定问题。优先用题目 (claim, 显式参数 → contextvar 注入),
    都没有才退回检索关键词 query。返回 (question, question_origin) 供日志。
    """
    claim = (claim or "").strip() or _active_claim()
    if claim:
        return claim, "claim"
    return (query or "").strip(), "query"


def filter_rag_hits_by_evidence(
    query: str,
    hits: list[dict[str, Any]],
    text_key: str = "text",
    claim: str = "",
    collection: str = "",
    image_driven: bool = False,
) -> list[dict[str, Any]]:
    """对 RAG 命中列表做证据过滤: 仅保留被判定为「支撑性证据」的命中。

    整批开关 (image_driven):
    双路 RAG (radimagenet / multimodal) 是以图像为主要查询源的检索 —— 题目
    是影像病例, 检索用查询图像去 match 库内图像的语义, 命中的证据在图像
    本体, 拿文本 passage 做 LLM 判定必然误删 (日志里图像召回几乎全被判 No)。
    因此这类"图像驱动"检索整批放行, 不做过滤。只有纯文本(关键字)驱动的
    检索 (miriad 等) 才走 Evidence-filter 判定。

    Parameters
    ----------
    query : str
        RAG 检索关键词(当没有 claim 时作为 evidence model 的判定问题)。
    hits : list[dict]
        融合后的命中列表, 每个命中需含 text_key 对应的文本字段。
    text_key : str
        命中的文本字段名: multimodal/radimagenet 为 "text"; miriad 为 "answer"。
    claim : str
        题目文本(优先作为判定问题)。不传时使用 set_claims_for_filter()
        注入的当前样本题目; 两者都没有才退回 query。
    collection : str
        本次检索涉及的 Milvus collection 名(多个用 " + " 拼接), 仅记入
        过滤日志, 便于事后核对某批判定的命中来自哪张表。
    image_driven : bool
        本次检索是否以图像为查询源。True → 整批放行不做过滤 (证据在图像);
        False → 仅对纯文本召回做证据过滤。由各 rag tool 在调用时声明。

    Returns
    -------
    list[dict]
        过滤后的命中列表。任何异常 / 禁用 / 模型不可用 / image_driven →
        原样返回(不过滤); 单条无法判定(空文本/调用失败) → 该条保留。
    """
    if not hits or not _filter_enabled():
        return hits
    if image_driven:
        logger.info(
            "Evidence filter skipped: image-driven RAG (%d hits 直接放行)",
            len(hits),
        )
        return hits
    question, origin = _evidence_question(query, claim)
    if not question:
        return hits

    passages = [(h or {}).get(text_key) for h in hits]
    # 命中来源 (text 路 / image 路 / 两路都中): dual-channel RAG 的图像路
    # 召回与关键词无关, 记下来供日志说明"这条为什么和检索词不搭"
    rag_sources: list[Optional[list[str]]] = [
        (h or {}).get("sources") for h in hits
    ]
    labels: list[Optional[str]]
    try:
        if _backend() == "llm":
            # rag_query = 工具实际传给 RAG 检索的关键词(LLM 生成的),
            # 与判定问题(question, 一般是 claim)分开记录
            labels = _infer_labels_llm(question, passages, origin,
                                       rag_query=query, rag_sources=rag_sources,
                                       collection=collection)
        else:
            engine = _get_engine()
            if engine is None:
                return hits
            # 与 RAG 编码/检索共用全局门闸, 避免本地推理与其他 RAG 调用
            # 抢占显存。懒加载 import 打破循环依赖
            # (本模块被 multimodal_rag_tool 顶层 import)。
            with _INFER_LOCK:
                from src.tools.multimodal_rag_tool import _RAG_GATE
                with _RAG_GATE:
                    labels = _infer_labels_local(question, passages, engine, origin,
                                                 rag_sources=rag_sources)
    except Exception:
        logger.exception(
            "Evidence filter inference failed; returning hits unfiltered (fail-open)."
        )
        return hits

    # 三态判定: 只有模型明确 No 才删除; None(空文本/调用失败)/乱答/Yes 全保留
    kept = [h for h, lab in zip(hits, labels) if not _is_no(lab)]
    removed = len(hits) - len(kept)
    if removed:
        empty = sum(1 for lab in labels if not (lab or "").strip())
        logger.info(
            "Evidence filter removed %d/%d hits for question(%s)=%r "
            "(empty labels=%d)",
            removed, len(hits), origin, question[:80], empty,
        )
    else:
        logger.info(
            "Evidence filter kept all %d hits for question(%s)=%r",
            len(hits), origin, question[:80],
        )
    return kept


def preload_rag_evidence_filter() -> Optional[Any]:
    """启动时预加载当前后端(可选): llm → 建好 OpenAI 客户端;
    local → 加载本地模型。被禁用 / 初始化失败返回 None, 不抛异常。
    """
    if not _filter_enabled():
        logger.info("Evidence filter disabled (RAG_EVIDENCE_FILTER != 1).")
        return None
    if _backend() == "llm":
        try:
            return _get_llm_client()
        except Exception:
            logger.exception("Evidence filter LLM client init failed.")
            return None
    return _get_engine()
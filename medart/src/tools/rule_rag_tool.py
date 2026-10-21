"""
Rule-based RAG tool for LangChain agent.

从 final_rules.json 中根据 LLM 给出的关键词（keyword）做粗粒度检索：
- 输入：keyword（必选，LLM 从病例/claim 中提取的概念关键词）
- 输出：命中的规则条目（rule / keywords / scenario / checkpoint 等）
- 数据源：outputs_medical/final_rules.json（本地 JSON 文件，懒加载缓存）

匹配语义（默认）：
- keyword 按中英文逗号/分号切分为若干词元（token）
- 一条规则至少命中 2 个词元（每个词元在其任一文本字段——含中英文字段、
  keywords 数组逐元素展开——中命中）才被召回
- 命中词元数越多排越前；同分保持规则文件顺序（稳定排序）
- 大小写不敏感

LLM 相关度精筛（RULE_FILTER=0 可关闭）：
- 用 case_id 从 input/recognition_export.json 读取题目(claim)
- 一次批量调用 tool 内置的小模型判断每条规则与题目是否相关
- 只保留相关且置信度 >= 阈值(默认 0.5，RULE_FILTER_CONFIDENCE 可调)的规则，其余不返回
- LLM 调用/解析失败或 claim 找不到时整体不返回规则(宁可错杀，避免误导主模型)

遵循项目 tool 约定（同 miriad_rag_tool.py / multimodal_rag_tool.py）：
- @tool + Pydantic args_schema
- 返回 (content_blocks, result) 元组：content_blocks 含文本块，多模态 LLM 可直接读；
  result 为应用层元数据
- 暴露纯函数 run_rule_rag_search()，不依赖 langchain，可被流水线直接调用

注册到 agent：在 run_multi_agent.py 中
    from src.tools.rule_rag_tool import retrieve_rule
    AVAILABLE_TOOLS = [..., retrieve_rule]
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import traceback
from typing import Any, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("rule_rag_tool")

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

# 默认规则文件路径（相对于项目根目录）
_DEFAULT_RULES_PATH = os.path.join("corrector_v2/outputs", "final_rules.json") # 大数据量跑出来的

# 支持通过环境变量覆盖
RULES_FILE_PATH = os.environ.get("RULE_RAG_FILE", _DEFAULT_RULES_PATH)

# ---------------------------------------------------------------------------
# LLM 相关度精筛配置 —— 关键字检索后，用题目(claim)判断规则是否相关再返回
# ---------------------------------------------------------------------------
# 开关：RULE_FILTER=0/false/off/no/disabled 关闭精筛（直接返回关键字检索结果）
RULE_FILTER_ENABLED = (
    os.environ.get("RULE_FILTER", "1").strip().lower()
    not in ("0", "false", "off", "no", "disabled")
)

# 题目(claim)数据文件：按 case_id 匹配 id 字段读取题目原文
# 默认相对项目根目录向上查找；可用环境变量 RULE_CLAIM_FILE 覆盖
_DEFAULT_CLAIM_FILE = os.path.join("input", "recognition_export.json")
CLAIM_FILE_PATH = os.environ.get("RULE_CLAIM_FILE", _DEFAULT_CLAIM_FILE)

# 规则数据缓存（懒加载，线程安全）
_rules_cache: Optional[list[dict[str, Any]]] = None
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------

# keyword 切分正则：中英文逗号 / 顿号 / 分号
_KEYWORD_SPLIT_RE = re.compile(r"[,，、;；]+")

# 参与匹配的标量文本字段
_SCALAR_TEXT_FIELDS = (
    "rule",
    "rule_en",
    "scenario",
    "scenario_en",
    "checkpoint",
    "checkpoint_en",
    "error_behavior",
    "correct_reasoning",
)

# 参与匹配的列表文本字段（逐元素匹配）
_LIST_TEXT_FIELDS = (
    "keywords",
    "keywords_en",
)


def _iter_rule_texts(rule: dict[str, Any]) -> list[str]:
    """把一条规则的所有可检索文本字段(标量 + 列表元素)拍平成小写文本列表。"""
    texts: list[str] = []
    for field in _SCALAR_TEXT_FIELDS:
        value = rule.get(field)
        if isinstance(value, str) and value:
            texts.append(value.lower())
    for field in _LIST_TEXT_FIELDS:
        values = rule.get(field) or []
        if isinstance(values, (list, tuple)):
            for v in values:
                if isinstance(v, str) and v:
                    texts.append(v.lower())
    return texts


def _split_keyword(keyword: str) -> list[str]:
    """把 keyword 切成小写词元，去空白去空，保留去重。"""
    terms: list[str] = []
    seen: set[str] = set()
    for part in _KEYWORD_SPLIT_RE.split(keyword or ""):
        term = part.strip().lower()
        if not term:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


# ---------------------------------------------------------------------------
# 规则加载
# ---------------------------------------------------------------------------

def _resolve_rules_path() -> str:
    """解析规则文件绝对路径，从当前工作目录向上查找。"""
    if os.path.isabs(RULES_FILE_PATH) and os.path.exists(RULES_FILE_PATH):
        return RULES_FILE_PATH

    # 从 cwd 逐级向上查找
    cur = os.getcwd()
    while True:
        candidate = os.path.join(cur, RULES_FILE_PATH)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # 找不到就返回原始路径（调用处会报文件不存在）
    return RULES_FILE_PATH


def _load_rules() -> list[dict[str, Any]]:
    """懒加载规则 JSON 并缓存，线程安全。"""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    with _cache_lock:
        if _rules_cache is not None:
            return _rules_cache

        path = _resolve_rules_path()
        logger.info("Loading rules from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = data.get("rules", []) if isinstance(data, dict) else data
        _rules_cache = rules if isinstance(rules, list) else list(rules)
        logger.info("Loaded %d rules", len(_rules_cache))
        return _rules_cache


# ---------------------------------------------------------------------------
# 题目(claim)加载 —— 精筛判题需要 case_id 对应的题目原文
# ---------------------------------------------------------------------------

_claims_cache: Optional[list[dict[str, Any]]] = None
_claim_cache_lock = threading.Lock()


def _resolve_claims_path() -> str:
    """解析 claim 文件绝对路径，从当前工作目录向上查找（同 _resolve_rules_path）。"""
    if os.path.isabs(CLAIM_FILE_PATH) and os.path.exists(CLAIM_FILE_PATH):
        return CLAIM_FILE_PATH
    cur = os.getcwd()
    while True:
        candidate = os.path.join(cur, CLAIM_FILE_PATH)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return CLAIM_FILE_PATH


def _load_claims() -> list[dict[str, Any]]:
    """懒加载 claim 文件并缓存，线程安全。文件不存在/解析失败会抛异常（由调用方兜底）。"""
    global _claims_cache
    if _claims_cache is not None:
        return _claims_cache
    with _claim_cache_lock:
        if _claims_cache is not None:
            return _claims_cache
        path = _resolve_claims_path()
        logger.info("Loading claims from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _claims_cache = data if isinstance(data, list) else list(data)
        logger.info("Loaded %d claims", len(_claims_cache))
        return _claims_cache


def _load_claim_text(case_id: Optional[str]) -> str:
    """按 case_id 精确匹配 id 字段，返回题目文本(claim)；未找到返回空串。"""
    cid = (case_id or "").strip()
    if not cid:
        return ""
    for rec in _load_claims():
        rec_id = rec.get("id")
        if rec_id == cid or (isinstance(rec_id, str) and rec_id.endswith(cid)):
            claim = rec.get("claim")
            if isinstance(claim, str):
                return claim
    return ""


# ---------------------------------------------------------------------------
# LLM 相关度精筛 —— 一次批量调用 tool 内置的小模型，判题规则是否相关
# ---------------------------------------------------------------------------
# todo  修改的时候，这里需要修改。
# OpenAI 兼容 API（独立小模型，tool 内部懒加载自管；环境变量可覆盖默认值）
FILTER_LLM_API_KEY = os.environ.get(
    "RULE_FILTER_LLM_API_KEY",
"*****",#
)
FILTER_LLM_BASE_URL = os.environ.get(
    "RULE_FILTER_LLM_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/plan/v3",
)
FILTER_LLM_MODEL = os.environ.get("RULE_FILTER_LLM_MODEL", "doubao-seed-2.0-lite")
FILTER_CONFIDENCE_THRESHOLD = float(os.environ.get("RULE_FILTER_CONFIDENCE", "0.6"))

_filter_client: Optional[Any] = None
_filter_client_lock = threading.Lock()

_FILTER_JUDGE_INSTRUCTION = (
    "You are a medical QA reviewer. For the clinical question/claim below and "
    "each candidate medical rule, decide whether the rule is relevant to THIS "
    "question.\n\n"
    "Relevance criteria (in order of priority):\n"
    "1. MODALITY MATCH: If the question involves a specific imaging modality "
    "   (CT, MRI, ultrasound, X-ray, ECG, HSG, fundus photography, etc.), "
    "   a rule whose scenario uses a DIFFERENT modality should be marked "
    "   as irrelevant UNLESS the rule describes a disease mechanism or "
    "   error pattern that applies across modalities.\n"
    "2. ANATOMY MATCH: The rule's anatomy/organ system must match the question's.\n"
    "3. DISEASE/SCENARIO MATCH: The rule's error pattern or scenario must be "
    "   applicable to the question's clinical situation.\n\n"
    "Be strict: background medical knowledge that has no bearing on this "
    "specific question must be marked irrelevant.\n\n"
    "Return ONLY a JSON array, one object per rule, with exactly these fields:\n"
    '  {"index": <0-based rule number>, "relevant": true or false, '
    '"confidence": <0.0 to 1.0>, "reason": "<one short sentence>"}\n'
    "No extra text outside the JSON array."
)


def _get_filter_client() -> Any:
    """懒加载精筛小模型的 OpenAI 兼容客户端（只做配置初始化，不发网络请求）。"""
    global _filter_client
    if _filter_client is not None:
        return _filter_client
    with _filter_client_lock:
        if _filter_client is None:
            from openai import OpenAI

            _filter_client = OpenAI(api_key=FILTER_LLM_API_KEY, base_url=FILTER_LLM_BASE_URL)
            logger.info(
                "Rule filter LLM backend ready: model=%s base=%s",
                FILTER_LLM_MODEL, FILTER_LLM_BASE_URL,
            )
    return _filter_client


def _rel_label_true(v: Any) -> bool:
    """容忍模型返回 true/True/yes/1/相关 等形式的真值。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("yes", "true", "1", "y", "相关")


def _parse_filter_verdicts(text: str, n: int) -> Optional[list[dict[str, Any]]]:
    """把模型输出解析成与规则条数对齐的判定列表[第 i 条 --> dict]。

    解析失败返回 None，由调用方整体不返回规则(宁可错杀)；缺项判定视为不相关。
    """
    if not (text or "").strip():
        return None

    t = (text or "").strip()
    data: Any = None
    try:
        data = json.loads(t)
    except Exception:
        pass
    if data is None:
        start = t.find("{")
        if start >= 0:
            try:
                data = json.loads(t[start:])
            except Exception:
                data = None
    if isinstance(data, dict):
        # 容忍 {"results": [...]} 之类的包装
        for key in ("results", "rules", "verdicts"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return None

    verdicts: list[dict[str, Any]] = [{} for _ in range(n)]
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        if not isinstance(idx, int) or not (0 <= idx < n):
            continue
        verdicts[idx] = item
    for i in range(n):
        verdicts[i].setdefault("index", i)
        verdicts[i].setdefault("relevant", False)
        verdicts[i].setdefault("confidence", 0.0)
        verdicts[i].setdefault("reason", "")
    return verdicts


def _llm_filter_rules_by_claim(
    claim: str, hits: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """批量(一次调用)让精筛小模型判断每条规则与题目是否相关。

    返回 (保留的命中, 判定明细, 备注)。保留条件：relevant 为真且
    confidence >= FILTER_CONFIDENCE_THRESHOLD。任何调用/解析失败都返回
    (空列表, [], 错误说明) —— 调用方据此整体不返回规则(宁可错杀)。
    """
    if not hits:
        return [], [], ""

    rule_blocks: list[str] = []
    for i, h in enumerate(hits):
        rule_blocks.append(
            f"[Rule {i}]\n"
            f"  rule: {h.get('rule', '')}\n"
            f"  rule_en: {h.get('rule_en', '')}\n"
            f"  keywords: {', '.join(h.get('keywords', []) or [])}\n"
            f"  scenario: {h.get('scenario', '')}\n"
            f"  checkpoint: {h.get('checkpoint', '')}"
        )
    user_content = (
        f"## Question (claim)\n{claim}\n\n"
        f"## Candidate rules\n" + "\n----\n".join(rule_blocks)
    )

    try:
        client = _get_filter_client()
        resp = client.chat.completions.create(
            model=FILTER_LLM_MODEL,
            messages=[
                {"role": "system", "content": _FILTER_JUDGE_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Rule filter LLM call failed: %s", e)
        return [], [], f"rule filter LLM call failed: {type(e).__name__}: {e}"

    verdicts = _parse_filter_verdicts(text, len(hits))
    if verdicts is None:
        preview = text[:200].replace("\n", " ")
        return [], [], f"rule filter output unparsable, raw start: {preview!r}"

    kept: list[dict[str, Any]] = []
    for h, ver in zip(hits, verdicts):
        conf = ver.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        if _rel_label_true(ver.get("relevant")) and conf >= FILTER_CONFIDENCE_THRESHOLD:
            kept.append({
                **h,
                "relevance_confidence": conf,
                "relevance_reason": ver.get("reason", ""),
            })
    note = f"LLM relevance filter kept {len(kept)}/{len(hits)} rules"
    return kept, verdicts, note


# ---------------------------------------------------------------------------
# 检索核心
# ---------------------------------------------------------------------------

def _search_rules_by_keyword(keyword: str, top_k: int) -> tuple[list[dict[str, Any]], list[str]]:
    """按词元召回规则（至少命中 2 个词元才召回），返回 (命中列表, 匹配到的词元)。

    命中列表元素含原始规则字段 + matched_terms(命中词元) + match_count。
    """
    terms = _split_keyword(keyword)
    if not terms:
        return [], []

    # 命中词元集合（用于最终展示"哪些词元被命中"）
    matched_term_set: set[str] = set()
    scored: list[tuple[int, dict[str, Any]]] = []

    for rule in _load_rules():
        texts = _iter_rule_texts(rule)
        hit_terms = {t for t in terms if any(t in text for text in texts)}
        # 阈值语义：至少命中 2 个词元才召回该条
        if len(hit_terms) >= 2:
            matched_term_set.update(hit_terms)
            scored.append((len(hit_terms), {
                **rule,
                "matched_terms": sorted(hit_terms),
                "match_count": len(hit_terms),
            }))

    # 按命中词元数降序（稳定：同分保持文件顺序）
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [rule for _, rule in scored[:top_k]]
    return hits, sorted(matched_term_set)


# ---------------------------------------------------------------------------
# 纯函数入口（流水线可直接调用, 不依赖 langchain）
# ---------------------------------------------------------------------------

def run_rule_rag_search(
    keyword: str,
    case_id: Optional[str] = None,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按关键词从 final_rules.json 中检索相关规则，并经 LLM 相关度精筛后返回。

    Args:
        keyword: 检索关键词（必选），可含逗号分隔的多个概念词
        case_id: 样本 ID。用于从 claim 文件(input/recognition_export.json)
            读取题目文本，再判断检索到的规则是否与本题相关；精筛开启时必须有。
        top_k: 返回条数（默认 5）

    Returns:
        (content_blocks, result) 元组。精筛后只返回相关且置信度达标的规则；
        LLM 调用/解析失败或 claim 找不到时整体不返回规则(宁可错杀)。
    """

    def _error_return(msg: str, tb: Optional[str] = None):
        text = f"Rule RAG search failed: {msg}"
        if tb:
            text += f"\n\nFull traceback:\n{tb}"
        blocks = [{"type": "text", "text": text}]
        result = {
            "success": False,
            "case_id": case_id or "",
            "keyword": keyword or "",
            "total_hits": 0,
            "hits": [],
            "matched_terms": [],
            "warnings": [],
            "message": msg,
            "traceback": tb or "",
        }
        return blocks, result

    # 1. 输入校验
    keyword = (keyword or "").strip()
    if not keyword:
        return _error_return(
            "keyword is empty; pass non-empty search keywords "
            "(e.g. 'mandibular fracture, cleft jaw').")

    warnings: list[str] = []
    try:
        hits, matched_terms = _search_rules_by_keyword(keyword, top_k=10)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Rule RAG keyword search failed: keyword=%r", keyword)
        return _error_return(
            f"search error: {type(e).__name__}: {e}", tb=tb)

    if not matched_terms:
        return _error_return(
            f"No rule matched at least 2 keyword terms of '{keyword}'; "
            "provide 2+ comma-separated medical concepts.")

    # 2. LLM 相关度精筛：用题目(claim)一次性批量判断每条规则是否与本题相关
    if RULE_FILTER_ENABLED and hits:
        try:
            claim_text = _load_claim_text(case_id)
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("Rule filter claim load failed: case_id=%r %s", case_id, e)
            return _error_return(
                f"claim lookup failed for case_id={case_id!r}: {type(e).__name__}: {e}",
                tb=tb)
        if not claim_text:
            return _error_return(
                f"claim not found for case_id={case_id!r} in {CLAIM_FILE_PATH}; "
                "cannot run LLM relevance filter (fail-closed: no rules returned).")
        hits, _verdicts, filt_note = _llm_filter_rules_by_claim(claim_text, hits)
        if filt_note.startswith("rule filter"):
            return _error_return(filt_note)
        warnings.append(filt_note)
        if not hits:
            # 全部被精筛掉: 非错误, 明确告知主模型没有相关规则, 自由分析
            overview = [
                "## Rule RAG Results (source: final_rules.json)\n",
                f"- case_id: {case_id or '(none)'}",
                f"- keyword: {keyword}",
                "- Matched rules: 0 (no retrieved rule is relevant to this claim)",
                f"- Matched keyword terms: {', '.join(matched_terms) or '(none)'}\n",
            ]
            content_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": "\n".join(overview)},
            ]
            result: dict[str, Any] = {
                "success": True,
                "case_id": case_id or "",
                "keyword": keyword,
                "total_hits": 0,
                "matched_terms": matched_terms,
                "hits": [],
                "warnings": warnings,
                "message": "No retrieved rule passed the LLM relevance filter; "
                "do not reference any rule, analyze freely.",
            }
            return content_blocks, result
    elif hits:
        warnings.append("rule LLM relevance filter disabled by RULE_FILTER=0")

    # 按 top_k 截断：无论来自关键字检索还是精筛结果，最终至多返回 top_k 条
    hits = hits[:top_k]

    # 3. content_blocks: 概览 + 命中详情
    overview = [
        "## Rule RAG Results (source: final_rules.json)\n",
        f"- case_id: {case_id or '(none)'}",
        f"- keyword: {keyword}",
        f"- Matched rules: {len(hits)}",
        f"- Matched keyword terms: {', '.join(matched_terms) or '(none)'}\n",
    ]
    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(overview)},
    ]
    for i, h in enumerate(hits, 1):
        rule_lines = [
            f"- **Rule (EN)**: {h.get('rule_en', '')}",
            f"- **Keywords (EN)**: {', '.join(h.get('keywords_en', []))}",
            f"- **Scenario (EN)**: {h.get('scenario_en', '')}",
            f"- **Checkpoint (EN)**: {h.get('checkpoint_en', '')}",
            f"- **Error behavior**: {h.get('error_behavior', '')}",
            f"- **Matched terms** ({h.get('match_count', 0)}): "
            f"{', '.join(h.get('matched_terms', [])) or '(none)'}",
            f"- **Relevance confidence**: {h.get('relevance_confidence', 1.0):.2f} — "
        ]
        content_blocks.append({"type": "text", "text": "\n".join(rule_lines)})

    if warnings:
        content_blocks.append({
            "type": "text",
            "text": "**Warnings**:\n" + "\n".join(f"- {w}" for w in warnings),
        })

    # 4. result 元数据
    result: dict[str, Any] = {
        "success": True,
        "case_id": case_id or "",
        "keyword": keyword,
        "total_hits": len(hits),
        "matched_terms": matched_terms,
        "hits": [
            {
                "rank": i,
                "rule_en": h.get("rule_en", ""),
                "keywords_en": h.get("keywords_en", []),
                "scenario_en": h.get("scenario_en", ""),
                "checkpoint_en": h.get("checkpoint_en", ""),
                "error_behavior": h.get("error_behavior", ""),
                "matched_terms": h.get("matched_terms", []),
                "match_count": h.get("match_count", 0),
            }
            for i, h in enumerate(hits, 1)
        ],
        "warnings": warnings,
        "message": (
            f"Found {len(hits)} rule(s) matching keyword terms "
            f"[{', '.join(matched_terms)}]."
        ),
    }
    return content_blocks, result


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

class RuleRagSearchInput(BaseModel):
    keyword: str = Field(
        description=(
            "Required. Search keywords describing the medical concepts in "
            "the current case / claim, e.g. 'mandibular defect, fracture' or "
            "'pelvic avulsion, ischial tuberosity'. Separate distinct "
            "concepts with commas. Used to retrieve relevant pre-curated "
            "medical rules from final_rules.json. A rule is returned only if "
            "at least two of these terms appear in any of its text fields."
        ),
        min_length=1,
    )
    case_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional but recommended. The sample / case ID, used to look up "
            "the claim text (input/recognition_export.json) and filter "
            "retrieved rules by relevance to the question. Retrieval itself "
            "is based on 'keyword'; without case_id no claim can be loaded "
            "and no rules are returned when relevance filtering is enabled."
        ),
    )
    # top_k: Optional[int] = Field(
    #     default=1,
    #     description="Optional. Max number of matched rules to return (default 1).",
    # )


@tool(args_schema=RuleRagSearchInput)
def retrieve_rule(
    keyword: str,
    case_id: Optional[str] = None,

) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    如果是诊断类、治疗类，必须调用这个tool.！既往根据大模型常见错误，而整理的诊断、治疗规则，非治疗、诊断类，请勿使用本规则。**若未命中任何相关规则，请直接基于图像与临床信息、结合自身专业知识进行推理，不要改调其他检索工具（尤其不要调用 retrieve_unified_knowledge）**。可基于图像与临床信息自由分析，**请勿绝对化应用规则，这些规则仅是过去的错题整理，还需要参考其他的选项、知识***：
    - rule / rule_en：规则正文（中/英）
    - keywords / keywords_en：关键词列表
    - scenario / scenario_en：适用场景
    - checkpoint / checkpoint_en：检查点
    - error_behavior / correct_reasoning：错误行为与正确推理路径

    检索方式：将 keyword 按逗号切分为词元，在每条规则的文本字段中做
    大小写不敏感的子串匹配，至少命中 2 个词元才召回该条；命中词元数越多越靠前。
    检索后会用 case_id 读取题目(claim)，调用 tool 内置的小模型一次批量判断
    每条规则是否与本题相关，只返回相关且置信度达标的规则。
    case_id 用于读取题目并参与精筛，不参与关键字匹配。

    Inputs:
    - keyword: 检索关键词（必选），逗号分隔多个医学概念
    - case_id: 病例/样本 ID，用于读取题目(claim)并做相关度精筛

    Outputs (content_blocks → visible to LLM):
    - Text: 概览 + 命中的规则详情（rule、keywords、scenario、checkpoint、
      error_behavior、correct_reasoning 等中英文对照 + 命中词元）

    Outputs (result dict → application metadata):
    - success / total_hits / matched_terms
    - hits: [{rank,  rule, rule_en, keywords, ..., matched_terms,
      match_count}]
    - warnings: 非致命提示
    - message: 状态描述
    """
    top_k= 1
    return run_rule_rag_search(
        keyword=keyword, case_id=case_id, top_k=top_k )


# 兼容命名: 与其他 rag tool 的 preload 语义对齐
def preload_rule_models() -> None:
    """启动前预加载规则文件并常驻内存，避免首次调用现场加载。"""
    _load_rules()
    logger.info("Rule RAG rules preloaded from: %s", RULES_FILE_PATH)

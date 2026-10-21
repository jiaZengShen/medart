"""
Debug tracker for SingleAgentV1 — captures all intermediate state for analysis.

Usage:
    tracker = DebugTracker(debug_dir="debug_reports")
    agent = SingleAgentV1(..., debug_tracker=tracker)
    result = agent.run(user_input)
    tracker.generate_html()  # Creates HTML report with pagination + filtering

This module captures:
  1. _call_biomedparse_for_side_verification output (side, mask image, analysis)
  2. _detect_with_llm output (LLM detection result)
  3. Main agent final_text + parsed answer
  4. Verifier agent final_text + parsed answer
  5. Original claim, context, image_paths
  6. Full raw LLM text outputs

Images use local file:/// URLs (NOT base64) to keep the HTML small.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote


# 样本卡片循环配色（浅色背景 + 强调色），按样本序号分配，
# 筛选/翻页后颜色不变，便于在报告中区分相邻样本。
CARD_PALETTE = [
    ("#fff8ec", "#d98e04"),  # 琥珀
    ("#eef6ff", "#2f6fd0"),  # 蓝
    ("#eefbef", "#2e9e4f"),  # 绿
    ("#fdf0f6", "#c23a6b"),  # 粉
    ("#f4f0ff", "#6a4fc3"),  # 紫
    ("#fffdee", "#a98f00"),  # 黄
    ("#eefafb", "#15878f"),  # 青
    ("#fff3ee", "#d1603a"),  # 橙
]


class DebugTracker:
    """Captures all intermediate state from SingleAgentV1 execution."""

    def __init__(self, debug_dir: str = "debug_reports") -> None:
        self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()
        self._thread_local = threading.local()

        # Per-sample snapshots
        self._snapshots: list[dict[str, Any]] = []

    # ── Snapshot management (thread-safe) ───────────────────────────────

    def new_snapshot(self, case_id: str) -> dict[str, Any]:
        """Start a new snapshot for one inference sample (thread-safe)."""
        snap = self._make_snapshot(case_id)
        with self._lock:
            self._snapshots.append(snap)
        self._thread_local.current_snapshot = snap
        return snap

    def _make_snapshot(self, case_id: str) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "claim": "",
            "context": "",
            "claim_type": "",
            "image_paths": [],
            "detect_with_llm": None,
            "biomedparse_side_analysis": None,
            "biomedparse_mask_path": None,
            "biomedparse_segmentation_prompt": None,
            "biomedparse_side_analysis_raw": None,
            "main_agent_input_text": "",
            "main_agent_final_text": "",
            "main_agent_parsed_answer": None,
            "main_agent_full_messages": None,
            "verifier_input_text": "",
            "verifier_final_text": "",
            "verifier_parsed_answer": None,
            "verifier_full_messages": None,
            "verifier_triggered_by": [],
            "original_answer": None,
            "answer_changed": None,
            "part_grounding": None,
            "confidence_triage": None,
            "errors": [],
            "final_answer": None,
            "confidence": None,
            "latency_s": None,
            "status": None,
            "true_label": None,
            # RAG 调用记录（LLM 请求参数 + 返回文本/图像）
            "rag_calls": [],
            # Multi-agent fields
            "agent_results": None,
            "final_decision": None,
        }

    @property
    def current(self) -> dict[str, Any]:
        try:
            return self._thread_local.current_snapshot
        except AttributeError:
            return self.new_snapshot("unknown")

    # ── Capture points ──────────────────────────────────────────────────

    def capture_basics(self, user_input: dict[str, Any]) -> None:
        """Capture claim, context, image_paths (no base64)."""
        snap = self.current
        snap["claim"] = user_input.get("agent_prompt") or user_input.get("claim") or ""
        snap["context"] = user_input.get("context") or ""
        snap["claim_type"] = user_input.get("claim_type") or "recognition"
        snap["image_paths"] = user_input.get("image_paths") or []
        snap["true_label"] = user_input.get("true_label")

    def capture_detect_with_llm(self, result: dict) -> None:
        self.current["detect_with_llm"] = dict(result)

    def capture_biomedparse_side_analysis(
        self,
        analysis_text: str | None,
        mask_path: str | None = None,
        segmentation_prompt: str | None = None,
        side_analysis_raw: dict | None = None,
    ) -> None:
        snap = self.current
        snap["biomedparse_side_analysis"] = analysis_text
        snap["biomedparse_side_analysis_raw"] = side_analysis_raw
        snap["biomedparse_segmentation_prompt"] = segmentation_prompt
        if mask_path:
            snap["biomedparse_mask_path"] = mask_path

    def capture_part_grounding(
        self,
        triggered: bool,
        parts_annotated: list[dict] | None = None,
        annotated_image_paths: dict | None = None,
    ) -> None:
        """Capture v6 part-grounding info (drawn boxes + annotated images)."""
        self.current["part_grounding"] = {
            "triggered": triggered,
            "parts_annotated": parts_annotated or [],
            "annotated_image_paths": annotated_image_paths or {},
        }

    def capture_confidence_triage(
        self,
        assessed_confidence: float,
        threshold: float,
        routed_to: str,
        assessment_text: str = "",
        source: str = "llm",
    ) -> None:
        """Capture v7 confidence-triage routing (预判置信度 → 路由路径).

        routed_to: "direct" (fast path) or "decompose_merge" (complex CoT).
        source: "llm" (LLM triage) or "fake" (offline-label experiment).
        """
        self.current["confidence_triage"] = {
            "assessed_confidence": assessed_confidence,
            "threshold": threshold,
            "routed_to": routed_to,
            "assessment_text": assessment_text,
            "source": source,
        }

    def capture_main_agent_result(
        self,
        final_text: str,
        parsed_answer: dict | None,
        input_text: str = "",
        full_messages: list | None = None,
    ) -> None:
        snap = self.current
        snap["main_agent_final_text"] = final_text
        snap["main_agent_parsed_answer"] = dict(parsed_answer) if parsed_answer else None
        snap["main_agent_full_messages"] = self._summarize_messages(full_messages)
        if input_text:
            snap["main_agent_input_text"] = input_text

    def capture_verifier_result(
        self,
        final_text: str,
        parsed_answer: dict | None,
        input_text: str = "",
        full_messages: list | None = None,
        triggered_by: list[str] | None = None,
    ) -> None:
        snap = self.current
        snap["verifier_final_text"] = final_text
        snap["verifier_parsed_answer"] = dict(parsed_answer) if parsed_answer else None
        snap["verifier_full_messages"] = self._summarize_messages(full_messages)
        if input_text:
            snap["verifier_input_text"] = input_text
        if triggered_by:
            snap["verifier_triggered_by"] = triggered_by

    def capture_final_result(self, final_output: dict) -> None:
        snap = self.current
        snap["original_answer"] = final_output.get("original_answer")
        snap["answer_changed"] = final_output.get("answer_changed")
        snap["final_answer"] = final_output.get("final_answer")
        snap["confidence"] = final_output.get("confidence")
        snap["latency_s"] = final_output.get("latency_s")
        snap["status"] = final_output.get("status")

        # Auto-persist this sample to JSONL file immediately
        self._persist_snapshot(snap)

    def capture_multi_agent_results(
        self,
        agent_results: list[dict],
        final_decision: dict,
    ) -> None:
        """Capture results from multi-agent voting system."""
        snap = self.current
        snap["agent_results"] = agent_results
        snap["final_decision"] = final_decision

    def capture_rag_call(self, request_info: dict, result: dict) -> None:
        """记录一次 RAG 调用（LLM 请求参数 + 返回的文本/图像）。

        通过 multimodal_rag_tool.register_rag_observer() 注册本方法后,
        流水线里每次 RAG 调用结束会自动回调这里。

        ⚠️ 回调线程未必是 new_snapshot() 所在线程（工具可能在 worker /
        LangGraph 内部线程执行），thread-local 取不到 current_snapshot 时
        不能新建孤儿快照（否则数据既不归属真实样本、也不会被持久化）。
        此时按 request_info 里的 case_id 回退匹配最近一个同 case 快照。
        """
        snap = getattr(self._thread_local, "current_snapshot", None)
        if snap is None:
            cid = request_info.get("case_id")
            with self._lock:
                for s in reversed(self._snapshots):
                    if cid and s.get("case_id") == cid:
                        snap = s
                        break
            if snap is None:
                return
        if "rag_calls" not in snap:
            snap["rag_calls"] = []
        snap["rag_calls"].append({
            "request": request_info,
            "response": result,
        })

    def _persist_snapshot(self, snap: dict[str, Any]) -> None:
        """Append one snapshot to the JSONL file (thread-safe)."""
        jsonl_path = self._debug_dir / "samples.jsonl"
        line = json.dumps(snap, ensure_ascii=False)
        with self._lock:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ── File helpers ────────────────────────────────────────────────────

    @staticmethod
    def _path_to_file_url(path: str) -> str:
        """Convert a Windows/Unix absolute path to a file:/// URL."""
        p = PureWindowsPath(path)
        posix_str = p.as_posix()
        if ":" in posix_str:
            url = "file:///" + posix_str
        else:
            url = "file://" + posix_str
        return quote(url, safe=":/#")

    @staticmethod
    def _path_exists(path: str) -> bool:
        return Path(path).exists()

    @staticmethod
    def _summarize_messages(messages: list | None) -> list | None:
        if not messages:
            return None
        summary = []
        for msg in messages:
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or type(msg).__name__
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                text_preview = content[:200]
            elif isinstance(content, list):
                text_preview = f"[{len(content)} blocks]"
            else:
                text_preview = str(content)[:200]
            summary.append({"role": role, "content_preview": text_preview})
        return summary

    # ── HTML report generation ──────────────────────────────────────────

    def generate_html(self) -> str:
        """Generate a paginated HTML report from in-memory snapshots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self._debug_dir / f"debug_report.html"
        html = self._build_html()
        html_path.write_text(html, encoding="utf-8")
        print(f"\n{'='*70}")
        print(f"  报告已生成: {html_path}  ({len(self._snapshots)} 条样本)")
        print(f"{'='*70}")
        return str(html_path)

    def save_json_dump(self) -> str:
        """Save all snapshots as JSON for programmatic analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self._debug_dir / f"debug_dump_{timestamp}.json"
        dump_data = [{k: v for k, v in snap.items() if not k.endswith("_data_url")} for snap in self._snapshots]
        json_path.write_text(json.dumps(dump_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"调试 JSON 已保存: {json_path}")
        return str(json_path)

    # ── Standalone report from JSONL ────────────────────────────────────

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> "DebugTracker":
        """Load all snapshots from a samples.jsonl file.

        Usage:
            tracker = DebugTracker.from_jsonl("debug_reports/samples.jsonl")
            tracker.generate_html()
        """
        path = Path(jsonl_path)
        if not path.exists():
            print(f"⚠️ 数据文件不存在: {jsonl_path}")
            return cls(debug_dir=str(path.parent))

        tracker = cls(debug_dir=str(path.parent))
        snapshots = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshots.append(json.loads(line))

        tracker._snapshots = snapshots
        print(f"已加载 {len(snapshots)} 条样本数据: {jsonl_path}")
        return tracker

    @classmethod
    def generate_report_from_jsonl(cls, jsonl_path: str) -> str:
        """Load JSONL and generate HTML report in one call.

        Returns:
            Path to the generated HTML file.
        """
        tracker = cls.from_jsonl(jsonl_path)
        return tracker.generate_html()

    # ── Build HTML ──────────────────────────────────────────────────────

    def _build_html(self) -> str:
        """Build the full paginated HTML with Chinese UI."""
        samples = self._snapshots
        n = len(samples)

        correct_flags: list[bool | None] = [self._get_correctness(s) for s in samples]
        n_correct = sum(1 for f in correct_flags if f is True)
        n_wrong = sum(1 for f in correct_flags if f is False)
        changed = sum(1 for s in samples if s.get("answer_changed") is True)
        bp_used = sum(1 for s in samples if s.get("biomedparse_side_analysis"))
        ver_used = sum(1 for s in samples if s.get("verifier_final_text"))
        grounded = sum(1 for s in samples if (s.get("part_grounding") or {}).get("triggered"))
        triage_stats = [(s.get("confidence_triage") or {}) for s in samples]
        routed_direct = sum(1 for t in triage_stats if t.get("routed_to") == "direct")
        routed_cot = sum(1 for t in triage_stats if t.get("routed_to") == "decompose_merge")
        errors = sum(1 for s in samples if s.get("errors"))
        rag_used = sum(1 for s in samples if s.get("rag_calls"))

        cards_html = "\n".join(
            self._build_sample_card(i, s, correct_flags[i]) for i, s in enumerate(samples)
        )
        err_count = errors

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单智能体调试报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Microsoft YaHei','PingFang SC','Segoe UI',sans-serif; background:#f0f2f5; margin:0; padding:20px; color:#333; }}
  h1 {{ color:#1a1a2e; border-bottom:3px solid #16213e; padding-bottom:10px; }}
  .toolbar {{ background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.1); padding:14px 18px; margin-bottom:16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
  .toolbar label {{ font-size:14px; font-weight:600; color:#555; }}
  .filter-btn {{ padding:5px 16px; border-radius:20px; border:1px solid #ccc; cursor:pointer; font-size:13px; background:#fff; transition:all .2s; }}
  .filter-btn:hover {{ background:#eee; }}
  .filter-btn.active {{ background:#0f3460; color:#fff; border-color:#0f3460; }}
  .filter-btn.wrong.active {{ background:#dc3545; border-color:#dc3545; }}
  .filter-btn.correct.active {{ background:#28a745; border-color:#28a745; }}
  select {{ padding:4px 8px; border-radius:4px; border:1px solid #ccc; font-size:13px; }}
  .search-box {{ padding:6px 12px; border-radius:20px; border:1px solid #ccc; font-size:13px; width:280px; }}
  .search-box:focus {{ outline:none; border-color:#0f3460; box-shadow:0 0 0 2px rgba(15,52,96,.2); }}
  .stats-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; }}
  .stat-card {{ background:#fff; border-radius:8px; padding:10px 18px; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,.08); min-width:90px; }}
  .stat-card .val {{ font-size:22px; font-weight:bold; }}
  .stat-card .lbl {{ font-size:12px; color:#888; margin-top:2px; }}
  .sample-card {{ background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.08); padding:18px; margin-bottom:18px; display:none; }}
  .sample-card.visible {{ display:block; }}
  .sample-card.wrong {{ border-left:4px solid #dc3545; }}
  .sample-card.correct {{ border-left:4px solid #28a745; }}
  .sample-card.unknown {{ border-left:4px solid #ffc107; }}
  .sample-header {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px; }}
  .sample-header .id {{ font-size:15px; font-weight:bold; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px; font-weight:bold; }}
  .badge-ok {{ background:#d4edda; color:#155724; }}
  .badge-err {{ background:#f8d7da; color:#721c24; }}
  .badge-warn {{ background:#fff3cd; color:#856404; }}
  .badge-info {{ background:#d1ecf1; color:#0c5460; }}
  .badge-neutral {{ background:#e2e3e5; color:#383d41; }}
  .section {{ background:#fafafa; border:1px solid #eee; border-radius:6px; padding:12px 14px; margin-bottom:10px; }}
  .section-title {{ font-weight:600; font-size:13px; color:#333; margin-bottom:6px; }}
  table.detail {{ width:100%; border-collapse:collapse; margin:6px 0; }}
  table.detail td, table.detail th {{ padding:5px 8px; border:1px solid #ddd; text-align:left; vertical-align:top; font-size:12px; }}
  table.detail th {{ background:#f5f5f5; font-weight:600; width:110px; }}
  pre.code {{ background:#1e1e2e; color:#cdd6f4; padding:10px; border-radius:5px; overflow-x:auto; font-size:12px; line-height:1.5; max-height:350px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }}
  pre.code.small {{ max-height:120px; }}
  .toggle-btn {{ background:#0f3460; color:#fff; border:none; padding:3px 10px; border-radius:4px; cursor:pointer; font-size:11px; margin:3px 0; }}
  .toggle-btn:hover {{ background:#16213e; }}
  .collapsed {{ display:none; }}
  .answer-compare {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .answer-compare > div {{ flex:1; min-width:100px; padding:8px; border-radius:5px; text-align:center; font-size:13px; }}
  .ans-orig {{ background:#e8f4f8; border:2px solid #b8d4e8; }}
  .ans-ver {{ background:#f0f0f0; border:2px solid #ccc; }}
  .ans-final {{ background:#d4edda; border:2px solid #a3c9b5; }}
  .ans-final.changed {{ background:#fff3cd; border:2px solid #e8c84a; }}
  .error-box {{ background:#f8d7da; border:1px solid #f5c6cb; color:#721c24; padding:8px 10px; border-radius:5px; margin:6px 0; font-size:12px; }}
  .img-grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .img-card {{ background:#f5f5f5; border:1px solid #ddd; border-radius:4px; padding:6px; }}
  .img-card img {{ max-width:min(100%,380px); max-height:320px; }}
  .img-card .cap {{ font-size:11px; color:#888; margin-top:3px; word-break:break-all; }}
  /* RAG 检索调用：紧凑网格, 一行并排多张命中卡片 */
  .rag-call {{ border-bottom:1px dashed #e2e2e2; padding:6px 0; }}
  .rag-call:last-child {{ border-bottom:none; }}
  .rag-request {{ font-size:11px; color:#444; background:#fff8ec; border:1px solid #f0e0c0; border-radius:4px; padding:4px 8px; margin-bottom:5px; word-break:break-all; }}
  .rag-grid {{ display:flex; flex-direction:column; gap:6px; }}
  .rag-card {{ display:flex; flex-direction:row; border:1px solid #ddd; border-radius:4px; background:#fff; overflow:hidden; }}
  .rag-card img {{ width:110px; min-width:110px; height:110px; object-fit:cover; display:block; }}
  .rag-nothumb {{ width:110px; min-width:110px; height:110px; background:#eee; color:#999; display:flex; align-items:center; justify-content:center; font-size:11px; }}
  .rag-cap {{ flex:1; font-size:13px; color:#333; padding:8px 10px; line-height:1.55; word-break:break-word; white-space:pre-wrap; }}
  .pagination {{ display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:4px; margin:16px 0; }}
  .pagination button {{ padding:5px 12px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:12px; }}
  .pagination button:hover:not(:disabled) {{ background:#eee; }}
  .pagination button.active {{ background:#0f3460; color:#fff; border-color:#0f3460; }}
  .pagination button:disabled {{ opacity:.4; cursor:default; }}
  .pagination .info {{ font-size:12px; color:#666; padding:0 8px; }}
</style>
</head>
<body>

<h1>🧪 单智能体调试报告</h1>
<p>生成时间: {datetime.now().isoformat()} | 样本总数: {n}</p>

<!-- ── 统计概览 ── -->
<div class="stats-row">
  <div class="stat-card"><div class="val" style="color:#0f3460;">{n}</div><div class="lbl">总样本</div></div>
  <div class="stat-card"><div class="val" style="color:#28a745;">{n_correct}</div><div class="lbl">正确</div></div>
  <div class="stat-card"><div class="val" style="color:#dc3545;">{n_wrong}</div><div class="lbl">错误</div></div>
  <div class="stat-card"><div class="val" style="color:#e94560;">{changed}</div><div class="lbl">答案变更</div></div>
  <div class="stat-card"><div class="val" style="color:#0f3460;">{bp_used}</div><div class="lbl">BiomedParse</div></div>
  <div class="stat-card"><div class="val" style="color:#0f3460;">{ver_used}</div><div class="lbl">核验者</div></div>
  <div class="stat-card"><div class="val" style="color:#e94560;">{grounded}</div><div class="lbl">部位标注</div></div>
  <div class="stat-card"><div class="val" style="color:#28a745;">{routed_direct}</div><div class="lbl">Direct路径</div></div>
  <div class="stat-card"><div class="val" style="color:#e94560;">{routed_cot}</div><div class="lbl">CoT路径</div></div>
  <div class="stat-card"><div class="val" style="color:{'#721c24' if err_count else '#28a745'};">{err_count}</div><div class="lbl">报错</div></div>
  <div class="stat-card"><div class="val" style="color:#15878f;">{rag_used}</div><div class="lbl">RAG调用</div></div>
</div>

<!-- ── 工具栏 ── -->
<div class="toolbar">
  <label>🔍 筛选:</label>
  <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">全部 ({n})</button>
  <button class="filter-btn correct" data-filter="correct" onclick="setFilter('correct')">✅ 正确 ({n_correct})</button>
  <button class="filter-btn wrong" data-filter="wrong" onclick="setFilter('wrong')">❌ 错误 ({n_wrong})</button>
  <span style="flex:1;"></span>
  <label>搜索:</label>
  <input type="text" id="searchInput" class="search-box" placeholder="搜索 ID 或声明..." oninput="setSearch(this.value)">
  <label>每页:</label>
  <select id="pageSizeSelect" onchange="setPageSize(this.value)">
    <option value="10">10</option>
    <option value="20" selected>20</option>
    <option value="50">50</option>
    <option value="100">100</option>
  </select>
</div>

<!-- ── 分页 (上) ── -->
<div class="pagination" id="paginationTop"></div>

<!-- ── 样本卡片容器 ── -->
<div id="cardsContainer">{cards_html}</div>

<!-- ── 分页 (下) ── -->
<div class="pagination" id="paginationBottom"></div>

<script>
const samples = document.querySelectorAll('.sample-card');
let currentFilter='all', currentPage=1, pageSize=20, searchQuery='';

function getVisibleIndices() {{
  const idx=[];
  const q = searchQuery.toLowerCase().trim();
  samples.forEach((el,i)=>{{
    const c=el.dataset.correct;
    let ok=true;
    if(currentFilter==='correct') ok=c==='true';
    else if(currentFilter==='wrong') ok=c==='false';
    if(ok && q) {{
      const idText = el.querySelector('.id')?.textContent?.toLowerCase() || '';
      const claimText = el.querySelector('.section-title + table.detail pre.code')?.textContent?.toLowerCase() || '';
      ok = idText.includes(q) || claimText.includes(q);
    }}
    if(ok) idx.push(i);
  }});
  return idx;
}}

function render() {{
  const v=getVisibleIndices(), tp=Math.max(1,Math.ceil(v.length/pageSize));
  if(currentPage>tp) currentPage=tp;
  const s=(currentPage-1)*pageSize, e=s+pageSize, pi=v.slice(s,e);
  samples.forEach(el=>el.classList.remove('visible'));
  pi.forEach(i=>samples[i].classList.add('visible'));
  renderPag('paginationTop',tp,v.length);
  renderPag('paginationBottom',tp,v.length);
}}

function renderPag(id,tp,tv) {{
  const el=document.getElementById(id);
  let h='<button onclick="goPage(currentPage-1)"'+(currentPage<=1?'disabled':'')+'>◀ 上一页</button>';
  let ps=Math.max(1,currentPage-4), pe=Math.min(tp,currentPage+5);
  if(ps>1) h+='<button onclick="goPage(1)">1</button>';
  if(ps>2) h+='<span style="color:#999;">…</span>';
  for(let p=ps;p<=pe;p++) h+='<button onclick="goPage('+p+')"'+(p===currentPage?' class=active':'')+'>'+p+'</button>';
  if(pe<tp-1) h+='<span style="color:#999;">…</span>';
  if(pe<tp) h+='<button onclick="goPage('+tp+')">'+tp+'</button>';
  h+='<button onclick="goPage(currentPage+1)"'+(currentPage>=tp?'disabled':'')+'>下一页 ▶</button>';
  h+='<span class="info">第 '+currentPage+'/'+tp+' 页，共 '+tv+' 条</span>';
  el.innerHTML=h;
}}

function goPage(p) {{
  const tp=Math.max(1,Math.ceil(getVisibleIndices().length/pageSize));
  p=Math.max(1,Math.min(p,tp)); if(p===currentPage) return;
  currentPage=p; render(); window.scrollTo({{top:0,behavior:'smooth'}});
}}

function setFilter(f) {{
  currentFilter=f; currentPage=1;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.filter-btn[data-filter="'+f+'"]').classList.add('active');
  render();
}}

function setSearch(q) {{
  searchQuery=q; currentPage=1; render();
}}

function setPageSize(s) {{ pageSize=parseInt(s); currentPage=1; render(); }}
function toggleText(id) {{ const el=document.getElementById(id); if(el) el.classList.toggle('collapsed'); }}
render();
</script>
</body>
</html>"""

    # ── Sample card builder ────────────────────────────────────────────

    def _build_sample_card(self, index: int, snap: dict[str, Any], is_correct: bool | None) -> str:
        """Build one sample card as a <div> with all debug info."""
        cid = snap.get("case_id", "unknown")
        status = snap.get("status", "?")

        correct_data = "unknown"
        if is_correct is True: correct_data = "true"
        elif is_correct is False: correct_data = "false"

        label_cls = {"success":"badge-ok","error":"badge-err","pipeline_error":"badge-err"}.get(status,"badge-neutral")
        badges = [f'<span class="badge {label_cls}">{status}</span>']
        if is_correct is True: badges.append('<span class="badge badge-ok">✅ 正确</span>')
        elif is_correct is False: badges.append('<span class="badge badge-err">❌ 错误</span>')
        else: badges.append('<span class="badge badge-neutral">❓ 未知</span>')

        if snap.get("answer_changed") is True:
            badges.append('<span class="badge badge-warn">答案变更</span>')
        elif snap.get("answer_changed") is False and snap.get("verifier_final_text"):
            badges.append('<span class="badge badge-info">核验一致</span>')
        if (snap.get("part_grounding") or {}).get("triggered"):
            badges.append('<span class="badge badge-info">🎯 部位标注</span>')
        triage = snap.get("confidence_triage") or {}
        if triage.get("routed_to") == "direct":
            badges.append(f'<span class="badge badge-ok">⚡ Direct (预判 {triage.get("assessed_confidence", 0):.2f})</span>')
        elif triage.get("routed_to") == "decompose_merge":
            badges.append(f'<span class="badge badge-warn">🔀 CoT (预判 {triage.get("assessed_confidence", 0):.2f})</span>')

        # ── 错误 ──
        err_html = ""
        if snap.get("errors"):
            err_html = f'<div class="error-box">{"".join(f"• {e}<br>" for e in snap["errors"])}</div>'

        # ── 图像 (file:// 路径，不含 base64) ──
        img_paths = snap.get("image_paths", [])
        img_html = ""
        if img_paths:
            cards = []
            for i, p in enumerate(img_paths):
                if self._path_exists(p):
                    cards.append(f'<div class="img-card"><img src="{self._path_to_file_url(p)}" alt="img{i}"><div class="cap">📁 {p}</div></div>')
                else:
                    cards.append(f'<div class="img-card" style="color:#999;">[文件不存在] {p}</div>')
            img_html = f'<div class="section"><div class="section-title">🖼️ 输入图像</div><div class="img-grid">{"".join(cards)}</div></div>'

        # ── 答案对比 ──
        orig_ans = snap.get("original_answer")
        ver_answer = snap.get("verifier_parsed_answer") or {}
        ver_ans_val = ver_answer.get("final_answer")
        final_ans_val = snap.get("final_answer")
        comp_html = ""
        if orig_ans is not None:
            fc = "ans-final changed" if snap.get("answer_changed") else "ans-final"
            comp_html = f'''
<div class="section"><div class="section-title">⚖️ 答案对比</div>
<div class="answer-compare">
  <div class="ans-orig"><strong>🤖 主智能体</strong><br><span style="font-size:20px;font-weight:bold;">{orig_ans}</span></div>
  <div class="ans-ver"><strong>🔄 核验者</strong><br><span style="font-size:20px;font-weight:bold;">{ver_ans_val or "—"}</span></div>
  <div class="{fc}"><strong>✅ 最终</strong><br><span style="font-size:20px;font-weight:bold;">{final_ans_val}</span></div>
</div></div>'''

        # ── BiomedParse 侧别验证 ──
        bp_text = snap.get("biomedparse_side_analysis")
        bp_path = snap.get("biomedparse_mask_path")
        bp_prompt = snap.get("biomedparse_segmentation_prompt")
        bp_raw = snap.get("biomedparse_side_analysis_raw")
        bp_html = ""
        if bp_text or bp_path:
            parts = []
            if bp_path and self._path_exists(bp_path):
                parts.append(f'<div class="img-card"><img src="{self._path_to_file_url(bp_path)}" alt="mask"><div class="cap">🎯 Mask: {bp_path}</div></div>')
            if bp_prompt:
                parts.append(f"<div><strong>分割提示:</strong> <code>{self._escape_html(bp_prompt)}</code></div>")
            if bp_text:
                parts.append(f"<pre class='code'>{self._escape_html(bp_text)}</pre>")
            if bp_raw:
                parts.append(f"<pre class='code small'>{self._escape_html(json.dumps(bp_raw, indent=2, ensure_ascii=False))}</pre>")
            bp_html = f'<div class="section"><div class="section-title">🔬 BiomedParse 侧别验证</div>{"".join(parts)}</div>'

        # ── _detect_with_llm ──
        llm_detect = snap.get("detect_with_llm")
        llm_html = ""
        if llm_detect:
            llm_html = f'''
<div class="section"><div class="section-title">🤖 _detect_with_llm 输出</div>
<table class="detail">
  <tr><th>has_side</th><td>{"✅ 是" if llm_detect.get("has_side") else "❌ 否"}</td></tr>
  <tr><th>has_negation</th><td>{"✅ 是" if llm_detect.get("has_negation") else "❌ 否"}</td></tr>
  <tr><th>推理</th><td><pre class="code small">{self._escape_html(llm_detect.get("reasoning",""))}</pre></td></tr>
</table></div>'''

        # ── 主智能体输出 ──
        main_text = snap.get("main_agent_final_text", "")
        main_answer = snap.get("main_agent_parsed_answer", {})
        main_html = ""
        if main_text:
            tid = f"m{index}"
            main_html = f'''
<div class="section"><div class="section-title">📝 主智能体输出</div>
{self._parsed_answer_table(main_answer)}
<p><button class="toggle-btn" onclick="toggleText('{tid}')">展开/收起完整文本</button></p>
<pre class="code small">{self._escape_html(main_text[:2000])}{"…" if len(main_text)>2000 else ""}</pre>
<pre id="{tid}" class="code collapsed">{self._escape_html(main_text)}</pre></div>'''

        # ── 核验者输出 ──
        ver_text = snap.get("verifier_final_text", "")
        ver_answer = snap.get("verifier_parsed_answer", {})
        ver_html = ""
        if ver_text:
            tid = f"v{index}"
            triggered = ", ".join(snap.get("verifier_triggered_by", []))
            ver_html = f'''
<div class="section"><div class="section-title">🔄 核验者输出</div>
<p style="font-size:11px;color:#888;">触发原因: {triggered or "N/A"}</p>
{self._parsed_answer_table(ver_answer)}
<p><button class="toggle-btn" onclick="toggleText('{tid}')">展开/收起完整文本</button></p>
<pre class="code small">{self._escape_html(ver_text[:2000])}{"…" if len(ver_text)>2000 else ""}</pre>
<pre id="{tid}" class="code collapsed">{self._escape_html(ver_text)}</pre></div>'''

        # ── 多智能体 V5 输出 ──
        agent_results = snap.get("agent_results")
        final_decision = snap.get("final_decision")
        v5_html = ""
        if agent_results and final_decision:
            tid = f"v5{index}"
            # 构建投票分布
            votes = final_decision.get("votes", {})
            vote_parts = []
            for ans, cnt in votes.items():
                vote_parts.append(f"<strong>{ans}</strong>: {cnt} 票")
            vote_summary = ", ".join(vote_parts)

            # 构建各智能体的输出
            agent_parts = []
            for agent_result in agent_results:
                if agent_result.get("success"):
                    agent_idx = agent_result.get("agent_idx", 0)
                    final_text = agent_result.get("final_text", "")
                    answer = agent_result.get("answer", {})
                    agent_tid = f"v5a{index}_{agent_idx}"
                    agent_parts.append(f'''
<div style="border:1px solid #ddd; padding:10px; margin:8px 0; border-radius:6px; background:#fafafa;">
  <h4 style="margin:0 0 8px 0; color:#333;">🤖 智能体 {agent_idx + 1} (答案: {answer.get("final_answer", "?")})</h4>
  {self._parsed_answer_table(answer)}
  <p><button class="toggle-btn" onclick="toggleText('{agent_tid}')">展开/收起完整推理</button></p>
  <pre class="code small">{self._escape_html(final_text[:2000])}{"…" if len(final_text)>2000 else ""}</pre>
  <pre id="{agent_tid}" class="code collapsed">{self._escape_html(final_text)}</pre>
</div>''')
                else:
                    agent_idx = agent_result.get("agent_idx", 0)
                    agent_parts.append(f'''
<div style="border:1px solid #fcc; padding:10px; margin:8px 0; border-radius:6px; background:#fff5f5;">
  <h4 style="margin:0 0 8px 0; color:#c33;">❌ 智能体 {agent_idx + 1} 执行失败</h4>
</div>''')

            v5_html = f'''
<div class="section"><div class="section-title">🗳️ Multi-Agent V5 多智能体投票</div>
<div style="background:#e8f4e8; padding:12px; border-radius:6px; margin-bottom:10px;">
  <strong>最终答案:</strong> <span style="font-size:18px; font-weight:bold;">{final_decision.get("final_answer", "?")}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>投票分布:</strong> {vote_summary or "—"}
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>平均置信度:</strong> {final_decision.get("confidence", "?")}
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>成功智能体:</strong> {final_decision.get("successful_agents", 0)}/{final_decision.get("total_agents", 0)}
</div>
<p><button class="toggle-btn" onclick="toggleText('{tid}')">展开/收起综合推理</button></p>
<pre class="code small">{self._escape_html(final_decision.get("reasoning", "")[:3000])}{"…" if len(final_decision.get("reasoning", ""))>3000 else ""}</pre>
<pre id="{tid}" class="code collapsed">{self._escape_html(final_decision.get("reasoning", ""))}</pre>
<div style="margin-top:12px; border-top:2px solid #ddd; padding-top:12px;">
  <h4 style="margin:0 0 10px 0;">各智能体独立推理:</h4>
  {"".join(agent_parts)}
</div>
</div>'''

        # ── V6 部位标注（part grounding）──
        grounding = snap.get("part_grounding") or {}
        grounding_html = ""
        if grounding.get("triggered") and grounding.get("parts_annotated"):
            parts = grounding["parts_annotated"]
            rows = []
            for info in parts:
                nb = info.get("normalized_box") or []
                pb = info.get("pixel_box") or []
                nb_str = "[" + ", ".join(f"{float(v):.3f}" for v in nb) + "]" if nb else "—"
                pb_str = "[" + ", ".join(str(v) for v in pb) + "]" if pb else "—"
                rows.append(
                    f"<tr><td>{info.get('image_index', '')}</td>"
                    f"<td>{self._escape_html(info.get('label') or '')}</td>"
                    f"<td>{nb_str}</td>"
                    f"<td>{pb_str}</td>"
                    f"<td>{self._escape_html(str(info.get('color', '')))}</td></tr>"
                )
            table_html = (
                "<table class='detail'>"
                "<tr><th>图像</th><th>部位</th><th>归一化框</th><th>像素框</th><th>颜色</th></tr>"
                + "".join(rows) + "</table>"
            )
            annotated_paths = grounding.get("annotated_image_paths") or {}
            img_cards = []
            for idx, p in sorted(annotated_paths.items(), key=lambda kv: str(kv[0])):
                if self._path_exists(p):
                    img_cards.append(
                        f'<div class="img-card"><img src="{self._path_to_file_url(p)}" alt="annotated{idx}">'
                        f'<div class="cap">🎯 Image {idx} 标注图: {p}</div></div>'
                    )
                else:
                    img_cards.append(f'<div class="img-card" style="color:#999;">[文件不存在] {p}</div>')
            imgs_html = f'<div class="img-grid">{"".join(img_cards)}</div>' if img_cards else ""
            grounding_html = (
                f'<div class="section"><div class="section-title">🎯 部位标注 (Part Grounding)</div>'
                f'{table_html}{imgs_html}</div>'
            )

        # ── V7 置信度路由（confidence triage）──
        # 始终展示该区块；字段缺失（旧数据 / 非 v7 智能体）时用 "--" 占位
        t_conf = triage.get("assessed_confidence")
        t_thr = triage.get("threshold")
        routed = triage.get("routed_to")
        if routed == "direct":
            route_label = "⚡ Direct（直接回答）"
        elif routed == "decompose_merge":
            route_label = "🔀 Decompose-Merge（拆解-作答-合并 CoT）"
        else:
            route_label = "--"
        conf_str = f"{t_conf:.2f}" if isinstance(t_conf, (int, float)) else "--"
        thr_str = f"{t_thr:.2f}" if isinstance(t_thr, (int, float)) else "--"
        assess_text = triage.get("assessment_text") or "--"
        t_source = triage.get("source")
        source_label = {
            "llm": "🤖 LLM 预判",
        }.get(t_source, t_source or "--")
        triage_html = f'''
<div class="section"><div class="section-title">🚦 置信度路由 (Confidence Triage)</div>
<table class="detail">
  <tr><th>预判置信度</th><td><strong style="font-size:14px;">{conf_str}</strong></td></tr>
  <tr><th>路由阈值</th><td>{thr_str}</td></tr>
  <tr><th>路由结果</th><td>{route_label}</td></tr>
  <tr><th>预判来源</th><td>{source_label}</td></tr>
  <tr><th>预判理由</th><td><pre class="code small">{self._escape_html(assess_text)}</pre></td></tr>
</table></div>'''

        # ── 基本信息 ──
        true_label = snap.get("true_label")
        claim_t = snap.get("claim_type","?")
        latency = snap.get("latency_s","?")
        claim = snap.get("claim","")
        ctx = snap.get("context","")

        # ── RAG 检索调用（请求参数 + 返回文本/图像, 紧凑网格）──
        rag_html = self._build_rag_section(index, snap)

        card_bg, card_accent = CARD_PALETTE[index % len(CARD_PALETTE)]
        return f'''
<div class="sample-card {correct_data}" data-correct="{correct_data}" data-index="{index}" style="background:{card_bg}; border-top:5px solid {card_accent};">
  <div class="sample-header">
    <span class="id" style="color:{card_accent};">🔹 样本 #{index+1}: {self._escape_html(cid)}</span>
    {" ".join(badges)}
  </div>
  {err_html}
  <div class="section"><div class="section-title">📋 基本信息</div>
  <table class="detail">
    <tr><th>样本 ID</th><td>{self._escape_html(cid)}</td></tr>
    <tr><th>推理类型</th><td>{claim_t}</td></tr>
    <tr><th>真实标签</th><td>{true_label if true_label is not None else "?"}</td></tr>
    <tr><th>耗时</th><td>{latency}s</td></tr>
    <tr><th>声明</th><td><pre class="code small">{self._escape_html(claim)}</pre></td></tr>
    <tr><th>上下文</th><td><pre class="code small">{self._escape_html(ctx)}</pre></td></tr>
  </table></div>
  {triage_html}
  {img_html}
  {rag_html}
  {grounding_html}
  {comp_html}
  {llm_html}
  {bp_html}
  {main_html}
  {ver_html}
  {v5_html}
</div>'''

    # ── RAG 检索调用区块 ───────────────────────────────────────────────

    def _build_rag_section(self, index: int, snap: dict[str, Any]) -> str:
        """构建「RAG 检索调用」区块。

        每个 RAG 调用显示一行请求参数摘要; 返回的命中用 140px 小卡片
        网格平铺 —— 一行可并排多张（缩略图 + 描述摘要）, 紧凑不稀松。
        """
        rag_calls = snap.get("rag_calls") or []
        if not rag_calls:
            return ""

        calls_html = []
        for ci, call in enumerate(rag_calls):
            req = call.get("request") or {}
            resp = call.get("response") or {}

            # ── 请求参数（一行摘要）──
            req_parts = [f"query=<b>{self._escape_html(req.get('query', ''))}</b>"]
            img = req.get("image_path", "")
            if img:
                req_parts.append(f"img=<code>{self._escape_html(Path(img).name)}</code>")
            if req.get("case_id"):
                req_parts.append(f"case=<b>{self._escape_html(str(req['case_id']))}</b>")
            if req.get("dataset"):
                req_parts.append(f"ds=<b>{self._escape_html(str(req['dataset']))}</b>")
            if req.get("top_k") is not None:
                req_parts.append(f"top_k={req['top_k']}")

            ok = resp.get("success")
            status_txt = "✅ 成功" if ok else ("❌ 失败" if ok is False else "—")
            req_line = (
                f"<b>#{ci + 1}</b> {status_txt} | "
                f"文本 {resp.get('n_text_hits', 0)} / 图像 {resp.get('n_image_hits', 0)} "
                f"/ 融合 {resp.get('total_hits', 0)} | {' '.join(req_parts)}"
            )

            # ── 返回命中（紧凑卡片网格, 一行多张）──
            cards = []
            for h in resp.get("hits") or []:
                img_path = h.get("image_path")
                text = (h.get("text") or "").strip()
                sources = "+".join(h.get("sources") or [])
                sim = h.get("similarity") or {}
                sim_txt = ", ".join(
                    f"{ch}={float(v):.2f}" for ch, v in sim.items() if v is not None
                ) or "—"
                if img_path and self._path_exists(img_path):
                    thumb = f'<img src="{self._path_to_file_url(img_path)}" alt="rag-hit">'
                else:
                    thumb = '<div class="rag-nothumb">无图</div>'
                cards.append(
                    f'<div class="rag-card">{thumb}'
                    f'<div class="rag-cap"><b>#{len(cards) + 1}</b> [{sources}] '
                    f'sim={sim_txt}<br>{self._escape_html(text)}</div></div>'
                )
            hits_html = f'<div class="rag-grid">{"".join(cards)}</div>' if cards else ""

            # ── 失败信息 ──
            err_html = ""
            if ok is False:
                msg = resp.get("message", "")
                tb = resp.get("traceback", "") or ""
                err_html = f'<div class="error-box">{self._escape_html(msg)}'
                if tb:
                    err_html += (
                        "<details><summary>traceback</summary>"
                        f'<pre class="code small">{self._escape_html(tb[:800])}</pre>'
                        "</details>"
                    )
                err_html += "</div>"

            calls_html.append(
                f'<div class="rag-call"><div class="rag-request">{req_line}</div>'
                f"{hits_html}{err_html}</div>"
            )

        return (
            f'<div class="section"><div class="section-title">🔎 RAG 检索调用 '
            f"({len(rag_calls)} 次)</div>{''.join(calls_html)}</div>"
        )

    # ── Shared helpers ──────────────────────────────────────────────────

    def _parsed_answer_table(self, answer: dict | None) -> str:
        if not answer:
            return "<p style='font-size:12px;color:#888'><em>无解析结果</em></p>"
        rsn = (answer.get("reasoning","") or "")[:1500]
        ellipsis = "…" if len(answer.get("reasoning","") or "") > 1500 else ""
        return f'''
<table class="detail">
  <tr><th>最终答案</th><td><strong style="font-size:14px;">{answer.get("final_answer","?")}</strong></td></tr>
  <tr><th>置信度</th><td>{answer.get("confidence","?")}</td></tr>
  <tr><th>推理摘要</th><td><pre class="code small">{self._escape_html(rsn)}{ellipsis}</pre></td></tr>
</table>'''

    @staticmethod
    def _escape_html(text: str | None) -> str:
        if text is None: return ""
        return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

    def _get_correctness(self, snap: dict) -> bool | None:
        pred = snap.get("final_answer")
        true_label = snap.get("true_label")
        if pred is None or true_label is None:
            return None
        ct = snap.get("claim_type", "")
        if isinstance(true_label, str) and ct == "medical":
            expected = ord(true_label.upper()) - ord("A")
            pred_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            p = pred_map.get(pred.strip().upper(), -1)
            return p == expected
        if isinstance(true_label, str):
            return (pred.lower() == "yes") == (true_label == "1")
        if isinstance(true_label, int):
            return (pred.lower() == "yes") == (true_label == 1)
        return None

"""
测试rag'
Single-agent v6 — Three-stage reasoning: Observe → Assess → Decide.

Based on v2 with ALL OBVISER mechanisms removed:
  - No OBVISER reflection recheck (no second LLM checking the output)
  - No laterality-consistency obviser (no image-flip blind re-judgment)

Kept from v2:
  - Correctly handle tool (content_blocks, result) return format
  - Support multi-round tool calls
  - Format-correction retry when the first answer parses to "Uncertain"
  - Debug tracker integration
  - No cache, clean and direct



  效果不错。
"""

from __future__ import annotations

import json
import re
import time
import traceback
from string import Template
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool as langchain_tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.llms import QuotaExceededError
from src.multi_agent.image_utils import build_image_data_url, is_image_too_small
from src.single_agent_v1.debug_tracker import DebugTracker

# [EVIDENCE-FILTER] 让 RAG 证据过滤用「题目」而不是检索关键词做判定
from src.tools.rag_evidence_filter import set_claims_for_filter


# RAG 检索工具列表 + 每次样本最多调用次数(含格式纠正重试轮)
RAG_TOOL_NAMES = (
    "retrieve_unified_knowledge",
)
MAX_RAG_CALLS = 2

# 自我反思置信度阈值：低于该值时触发一轮 self-reflection
SELF_REFLECTION_CONFIDENCE_THRESHOLD = 0.0  # 0 永远不触发， 1 永远触发。

# 自我反思的四个维度（英文表述）
SELF_REFLECTION_DIMENSIONS = [
    "Visual attention hijacked by a single feature — "
    "whether your judgment was dominated by one conspicuous finding "
    "while other important but subtle signs were overlooked.",
    "Misweighting of clinical manifestations — "
    "whether you assigned inappropriate weights to different clinical "
    "findings, overemphasizing minor signs while underweighting key ones.",
    "Replacing clinical judgment with statistical probability — "
    "whether you relied too heavily on base rates or common-case "
    "statistics instead of reasoning from the actual evidence in this case.",
    "Insufficient vigilance for malignant diseases — "
    "whether you were not sufficiently alert to features suggestive of "
    "malignancy or serious pathology, and prematurely dismissed them.",
]


# ==========================================================================
# V8-only prompts — identical to v6's three-stage framework, except that
# calling at least one RAG retrieval tool (multimodal / MIRIAD / RadImageNet)
# is a MANDATORY first step (Stage 0). Kept inside agent_v8.py so the
# shared prompts used by other agents stay untouched.
# ==========================================================================

V8_2_SYSTEM_PROMPT = """
Task: You are an experienced medical professional with expertise in analyzing various medical images and integrating them with clinical information to reach diagnostic conclusions or treatment recommendations. Given a medical image, patient clinical information (history, symptoms, examination findings), a question, and multiple choice options, select the single best answer.


## Workflow (strictly follow order, no skipping)

### 📚 Stage 0 (MANDATORY FIRST STEP): Retrieve Reference Material
1. you can call at least one RAG retrieval tool. Choose freely from:
  - `retrieve_unified_knowledge` — multimodal medical-imaging knowledge base (reference text + reference images). This knowledge base mainly contains medical imaging findings. If the Task Type is "medical", calling this tool is FORBIDDEN.— it is of no use and is harmful.
   - `retrieve_rule` — diagnosis and treatment rules compiled from common mistakes previously made by large models. If it returns no matching rule, reason directly using your own knowledge — do NOT call a different retrieval tool.
2. For each case, you may call retrieval tools AT MOST 2 times in total. Calls to any of the three tools above all count toward this limit. Use those calls to gather whatever reference evidence you need; afterwards you must proceed WITHOUT further retrieval. What is NOT allowed is calling zero retrieval tools.
3. Carefully read every returned reference text and examine every returned reference image. Treat them as background evidence for your later judgment.
4. **Do NOT skip this step** — not even when the case looks easy or when you think you can judge it by eye. The retrieval must happen first, every time.
5. If a tool reports no reference material for your query, note that and continue with the remaining stages using your own knowledge — do NOT try another retrieval tool.

### 🔍 Stage 1: Observe the Image
1. Carefully examine each provided medical image
2. Describe the key anatomical structures, signal characteristics, and morphological findings you see
3. Record only **clearly visible** findings in the image; do not imagine non-existent content
4. **Do not call any other tools** in this stage — use only your eyes and professional knowledge
5. **Pay attention to left/right side.** When describing findings, always specify which side of the patient they are on. Identify side markers (e.g., "L", "R") or use radiological convention (patient's left appears on the right side of axial images).
            - When viewing a coronal slice from the anterior (front) perspective:
            - Viewer’s left portion of the image = Patient’s right portion
            - Viewer’s right portion of the image = Patient’s left portion

### 📋 Stage 2: Read the Claim and Assess Tool Need
1. Carefully read the claim and clinical context
2. Compare the claim against both your Stage 1 observations and the Stage 0 reference material
3. **Tool Usage Guidelines**:

   ✅ **When you SHOULD use additional tools** (including more RAG retrieval):
   - The claim contains specific numerical measurements (e.g., head circumference in mm) requiring precise tool-based verification
   - The claim requires precise localization of a structure where boundaries are difficult to distinguish by naked eye
   - You **cannot** form a reasonable judgment through naked-eye observation plus the retrieved reference material

   ❌ **When you SHOULD NOT call additional tools** (vast majority of cases):
   - You have already called at least one RAG tool and can judge the claim as true or false through naked-eye observation + reference material
   - You just want "double confirmation" — this is not sufficient
   - You have already exhausted your 3 retrieval calls — do NOT call any retrieval tool again, judge with what you have

4. Available tools:
- `retrieve_unified_knowledge` — multimodal medical-imaging knowledge base (reference text + reference images). This knowledge base mainly contains medical imaging findings. For diagnosis- or treatment-related questions, calling this tool is FORBIDDEN — it is of no use and is harmful.
   - `retrieve_rule` — diagnosis and treatment rules compiled from common mistakes previously made by large models. If it returns no matching rule, reason directly using your own knowledge — do NOT call a different retrieval tool.
   - `measure_fetal_head_circumference`: Measure fetal head circumference, requires image_path and pixel_size_mm. MUST use for tasks involving fetal head circumference measurement.

5. **Pay special attention to side and negation.** When reading the claim:
   - **Side:** Compare the side mentioned in the claim with the side you observed in Stage 1. If they do not match, the claim is false regardless of other findings.
   - **Negation words:** Watch for words like "no", "not", "without", "absent", "free of", "no evidence of", "does not demonstrate", "lacks", or any phrase that indicates absence or denial. A claim with negation means the finding described is NOT present. Verify this against your Stage 1 observations.

### ✅ Stage 3: Decide
1. Synthesize the retrieved reference material + naked-eye observations + any additional tool results, and verify item by item against the claim
2. If the user's claim imposes an overly strict or absolute condition for "No" (e.g., "completely absent," "absolutely no evidence"), override it with your actual visual observation — answer "No" whenever there is any uncertainty or subtle finding, without needing to meet the claim's extreme standard.
3. Provide the final judgment

---


---

## Output Format

You MUST wrap your reasoning process with `<reasoning>` tags, then provide the final answer.

For Yes/No tasks:
<reasoning>
[Your reasoning process, including what the retrieved reference material said]
</reasoning>
Therefore, the final answer is: [Yes/No]

For multiple-choice questions:
<reasoning>
[Your reasoning process, including what the retrieved reference material said]
</reasoning>
Therefore, the final answer is: [A/B/C/D/E]

Do not output any other content.
"""

V8_HUMAN_TEMPLATE = Template("""Please analyze the following medical images and claim following the staged process.

## Case ID
$case_id


## Claim
$claim

## Clinical Context
$context

## Task Type
$claim_type

## Image Paths
$image_paths

---

Please begin your analysis. Remember:
1. Then observe the image, read the claim, and decide
2. Use the retrieved reference material as evidence in your judgment
""")

class SingleAgentV8_2:
    """Single-agent v6: Three-stage reasoning pipeline without OBVISER.

    Usage:
        agent = SingleAgentV6(llm=llm, tools=tools)
        result = agent.run(user_input)
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list,
        debug_tracker: DebugTracker | None = None,
    ) -> None:
        """
        Parameters
        ----------
        llm : ChatOpenAI
            The LLM to drive reasoning.
        tools : list
            List of available tools (will be wrapped to handle return format).
        debug_tracker : DebugTracker | None
            Optional debug tracker for capturing intermediate state.
        """
        self._llm = llm
        self._raw_tools = tools
        self._debug_tracker = debug_tracker

        # RAG 调用预算(每次样本运行开始时清零)
        self._rag_tool_names = set(RAG_TOOL_NAMES)
        self._max_rag_calls = MAX_RAG_CALLS
        self._rag_call_count = 0

        # Wrap tools to unify return format
        self._wrapped_tools = [self._wrap_tool(t) for t in tools]

        self._agent = create_agent(
            model=llm,
            tools=self._wrapped_tools,
            system_prompt=V8_2_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Tool wrappers — Unify handling of (content_blocks, result) format
    # ------------------------------------------------------------------

    def _wrap_tool(self, tool) -> Any:
        """Wrap LangChain tools to unify return format.

        Handles two types of tool returns:
          1. Normal tools: return content directly
          2. Segmentation tools: return (content_blocks, result) tuple
             - content_blocks: list containing image for LLM
             - result: structured result dict
        """
        original_name = getattr(tool, "name", "unknown_tool")
        original_func = tool.func if hasattr(tool, "func") else tool
        original_response_format = getattr(tool, "response_format", "content")

        @langchain_tool(
            name_or_callable=original_name,
            description=getattr(tool, "description", ""),
            args_schema=getattr(tool, "args_schema", None),
            response_format=original_response_format,
        )
        def wrapped_func(**kwargs: Any) -> Any:
            # 硬性 RAG 调用上限: 同一病例最多调用 _max_rag_calls 次检索工具,
            # 超出后不再真正调用, 直接提示模型基于现有信息判断。
            if original_name in self._rag_tool_names:
                if self._rag_call_count >= self._max_rag_calls:
                    return (
                        "Skills limit reached: this case has already made "
                        f"{self._max_rag_calls} retrieval (RAG) calls. Do NOT call any "
                        "retrieval tool again. Make your final judgment using the "
                        "reference material already retrieved plus your image observation."
                    )
                self._rag_call_count += 1
            try:
                result = original_func(**kwargs)
            except QuotaExceededError:
                raise  # 账号额度耗尽 → 冒泡到主程序整体停止
            except Exception as exc:
                traceback.print_exc()
                error_info = {
                    "success": False,
                    "error": str(exc),
                    "tool": original_name,
                    "parameters": dict(kwargs),
                }
                return json.dumps(error_info, ensure_ascii=False)

            # If tuple (content_and_artifact format)
            if isinstance(result, tuple) and len(result) == 2:
                content_blocks, artifact = result
                # content_blocks is already list[dict] with text and image_url
                return content_blocks

            # Normal format
            return result

        return wrapped_func

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, user_input: dict[str, Any]) -> dict[str, Any]:
        """Run three-stage reasoning on single sample.

        Parameters
        ----------
        user_input : dict
            Must include:
                "claim": str,
                "image_paths": list[str],
                "context": str (optional),
                "claim_type": str (optional),
                "case_id": str (optional),

        Returns
        -------
        dict with keys:
            "final_answer": ...,
            "confidence": ...,
            "reasoning": ...,
            "status": "success" | "error",
            "latency_s": ...,
        """
        start_time = time.time()
        pipeline_status = "success"

        # RAG 调用预算: 每次样本(含格式纠正重试轮)从 0 重新计数
        self._rag_call_count = 0

        try:
            # ── Build user message ────────────────────────────────────
            claim = user_input.get("agent_prompt") or user_input.get("claim", "")
            # [EVIDENCE-FILTER] 注入题目, 供 RAG 证据过滤作为判定问题
            set_claims_for_filter(user_input.get("claim", ""))
            context = user_input.get("context", "")
            claim_type = user_input.get("claim_type", "recognition")
            case_id = user_input.get("case_id", "unknown")
            image_paths = user_input.get("image_paths", [])

            # Debug: start snapshot & capture basics
            if self._debug_tracker is not None:
                self._debug_tracker.new_snapshot(case_id)
                self._debug_tracker.capture_basics(user_input)

            human_text = V8_HUMAN_TEMPLATE.substitute(
                case_id=case_id,
                claim=claim,
                context=context or "None",
                claim_type=claim_type,
                image_paths=json.dumps(image_paths, ensure_ascii=False),
            )

            # Build multi-modal user content (text + image base64)
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": human_text},
            ]
            for img_path in image_paths:
                try:
                    if is_image_too_small(img_path):
                        user_content.append({
                            "type": "text",
                            "text": f"[Image too small, skipped: {img_path}]",
                        })
                        continue
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": build_image_data_url(img_path)},
                    })
                except Exception:
                    traceback.print_exc()
                    user_content.append({
                        "type": "text",
                        "text": f"[Image read failed: {img_path}]",
                    })

            # ── First pass reasoning ─────────────────────────────────
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_content}]}
            )
            all_messages = result.get("messages", [])
            final_text = self._extract_final_text(all_messages)
            answer = self._parse_output(final_text, claim_type)

            # 保存初始(first-pass)答案，用于对比自我反思前后的变化
            initial_answer = {
                "final_answer": answer.get("final_answer", "Uncertain"),
                "confidence": answer.get("confidence", 0.0),
                "reasoning": answer.get("reasoning", ""),
            }

            # 纠正。
            # ── Format correction retry if parsed to Uncertain ───────
            if answer.get("final_answer") == "Uncertain" or answer.get("confidence", 0.5) < 0.6:
                # Build format correction message
                format_correction = self._build_format_correction_prompt(final_text, claim_type)
                correction_user_content: list[dict[str, Any]] = [
                    {"type": "text", "text": format_correction},
                ]
                # Add images again
                for img_path in image_paths:
                    try:
                        if is_image_too_small(img_path):
                            correction_user_content.append({
                                "type": "text",
                                "text": f"[Image too small, skipped: {img_path}]",
                            })
                            continue
                        correction_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": build_image_data_url(img_path)},
                        })
                    except Exception:
                        pass
                # Call agent again with format correction
                correction_result = self._agent.invoke(
                    {"messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": final_text},
                        {"role": "user", "content": correction_user_content}
                    ]}
                )
                correction_messages = correction_result.get("messages", [])
                correction_final_text = self._extract_final_text(correction_messages)
                correction_answer = self._parse_output(correction_final_text, claim_type)
                # Use corrected answer
                answer = correction_answer
                final_text = correction_final_text

            #反思
            # # ── Self-reflection when confidence is below threshold ───
            # if answer.get("confidence", 0.0) < SELF_REFLECTION_CONFIDENCE_THRESHOLD \
            #         and answer.get("final_answer") != "Uncertain":
            #     # Build self-reflection prompt
            #     reflection_prompt = self._build_self_reflection_prompt(
            #         final_text, claim_type,
            #         current_answer=answer.get("final_answer", ""),
            #         current_confidence=answer.get("confidence", 0.0),
            #     )
            #     reflection_user_content: list[dict[str, Any]] = [
            #         {"type": "text", "text": reflection_prompt},
            #     ]
            #     # Add images again
            #     for img_path in image_paths:
            #         try:
            #             if is_image_too_small(img_path):
            #                 reflection_user_content.append({
            #                     "type": "text",
            #                     "text": f"[Image too small, skipped: {img_path}]",
            #                 })
            #                 continue
            #             reflection_user_content.append({
            #                 "type": "image_url",
            #                 "image_url": {"url": build_image_data_url(img_path)},
            #             })
            #         except Exception:
            #             pass
            #     # Call agent again with self-reflection
            #     reflection_result = self._agent.invoke(
            #         {"messages": [
            #             {"role": "user", "content": user_content},
            #             {"role": "assistant", "content": final_text},
            #             {"role": "user", "content": reflection_user_content}
            #         ]}
            #     )
            #     reflection_messages = reflection_result.get("messages", [])
            #     reflection_final_text = self._extract_final_text(reflection_messages)
            #     reflection_answer = self._parse_output(reflection_final_text, claim_type)
            #     # Use reflected answer
            #     answer = reflection_answer
            #     final_text = reflection_final_text

            # Debug: capture main agent result
            if self._debug_tracker is not None:
                self._debug_tracker.capture_main_agent_result(
                    final_text=final_text,
                    parsed_answer=answer,
                    full_messages=all_messages,
                )

            # 收集所有阶段的消息，用于提取 rule_rag 调用记录
            all_stage_messages: list = list(all_messages)
            # if 'correction_messages' in locals():
            #     all_stage_messages.extend(correction_messages)
            # if 'reflection_messages' in locals():
            #     all_stage_messages.extend(reflection_messages)
            rule_rag_calls = self._extract_rule_rag_calls(all_stage_messages)

        except QuotaExceededError:
            raise  # 账号额度耗尽 → 冒泡到主程序整体停止
        except Exception as exc:
            traceback.print_exc()
            pipeline_status = "error"
            answer = {
                "final_answer": "Uncertain",
                "confidence": 0.0,
                "reasoning": f"Execution error: {exc}",
            }
            # 异常情况下初始答案也设为 Uncertain
            initial_answer = {
                "final_answer": "Uncertain",
                "confidence": 0.0,
                "reasoning": f"Execution error: {exc}",
            }

        elapsed = time.time() - start_time

        # rule_rag 调用记录 (异常情况下为空列表)
        rule_rag_calls_val: list[dict[str, Any]] = []
        if 'rule_rag_calls' in locals():
            rule_rag_calls_val = rule_rag_calls

        # Build final output dict
        final_output = {
            "final_answer": answer.get("final_answer", "Uncertain"),
            "confidence": answer.get("confidence", 0.0),
            "reasoning": answer.get("reasoning", ""),
            "status": pipeline_status,
            "latency_s": round(elapsed, 2),
            # 初始(first-pass)答案，记录自我反思/格式纠正之前的状态
            "initial_answer": initial_answer.get("final_answer", "Uncertain"),
            "initial_confidence": initial_answer.get("confidence", 0.0),
            "initial_reasoning": initial_answer.get("reasoning", ""),
            # rule_rag 工具调用记录：入参 (case_id/keyword) + 出参 (rule)
            "rule_rag_calls": rule_rag_calls_val,
        }

        # Debug: capture final result
        if self._debug_tracker is not None:
            self._debug_tracker.capture_final_result(final_output)

        return final_output

    # ------------------------------------------------------------------
    # Helper functions for parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rule_rag_calls(messages: list) -> list[dict[str, Any]]:
        """从 agent 返回的消息列表中提取 RAG 检索工具的调用记录。

        和 retrieve_rule 一样, retrieve_unified_knowledge 的调用同样登记进
        rule_rag_calls —— 两个工具完全相同的处理方式。

        每条记录包含:
            - tool_name: 工具名 ("retrieve_rule" 或 "retrieve_unified_knowledge")
            - tool_call_id: 调用 ID
            - input_args: 入参 (retrieve_rule: case_id/keyword; unified: query/image_path/top_k)
            - output_rule: 返回的 rule 文本 (retrieve_rule 命中时) 或
                           winner 概览 "winner=X score=Y" (unified 命中时)
            - output_hit_sample_id: retrieve_rule 返回的 sample_id 或 unified 的 winner 来源库
            - output_total_hits: 命中数
            - raw_output: 原始返回 (content_blocks 形式, 完整保留)

        Args:
            messages: langchain message 列表 (AIMessage + ToolMessage 交替)

        Returns:
            list of RAG call records
        """
        # 先收集所有 AIMessage 中的 tool_calls (id -> name + args)
        tool_call_map: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if tc_id and tc_name in ("retrieve_rule", "retrieve_unified_knowledge"):
                    tool_call_map[tc_id] = {
                        "tool_name": tc_name,
                        "tool_call_id": tc_id,
                        "input_args": tc_args or {},
                    }

        # 再收集所有 ToolMessage (id -> content)
        tool_result_map: dict[str, Any] = {}
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id in tool_call_map:
                tool_result_map[tc_id] = msg.content

        # 组装结果，解析返回中的命中信息(两个检索工具同样处理)
        results: list[dict[str, Any]] = []
        for tc_id, call_info in tool_call_map.items():
            raw_output = tool_result_map.get(tc_id)
            record = {
                **call_info,
                "output_rule": "",
                "output_hit_sample_id": "",
                "output_total_hits": 0,
                "raw_output": raw_output,
            }
            # 尝试从 content_blocks 中解析结果正文
            if isinstance(raw_output, list):
                if call_info["tool_name"] == "retrieve_unified_knowledge":
                    # retrieve_unified_knowledge: 概览头 "## Unified RAG Results
                    # (winner: X, score Y)" + 各来源统一的 "Fused results: N"
                    winner = ""
                    winner_score = ""
                    for block in raw_output:
                        if not isinstance(block, dict):
                            continue
                        text = block.get("text", "")
                        if not isinstance(text, str):
                            continue
                        if "## Unified RAG Results" in text and "winner:" in text:
                            m = re.search(r"winner:\s*(\w+),\s*score\s+([0-9.]+)", text)
                            if m:
                                winner = m.group(1)
                                winner_score = m.group(2)
                            break
                    record["output_rule"] = (
                        f"winner={winner} score={winner_score}" if winner else ""
                    )
                    record["output_hit_sample_id"] = winner
                else:
                    # retrieve_rule: 解析 Rule / sample_id / Matched rules
                    rule_text = ""
                    sample_id = ""
                    total_hits = 0
                    for block in raw_output:
                        if not isinstance(block, dict):
                            continue
                        text = block.get("text", "")
                        if not isinstance(text, str):
                            continue
                        # 提取 Rule 行
                        if "**Rule**:" in text:
                            for line in text.split("\n"):
                                if "**Rule**:" in line:
                                    rule_text = line.split("**Rule**:", 1)[1].strip()
                                    break
                        # 提取 sample_id
                        if "sample_id:" in text and "Rule #" in text:
                            for line in text.split("\n"):
                                if line.startswith("### Rule #") and "(sample_id:" in line:
                                    # 形如 "### Rule #1  (sample_id: MedXpertQA_MM-0116)"
                                    m = re.search(r"sample_id:\s*([^)\s]+)", line)
                                    if m:
                                        sample_id = m.group(1)
                                    break
                        # 提取 Matched rules 数
                        if "Matched rules:" in text:
                            for line in text.split("\n"):
                                if line.strip().startswith("- Matched rules:"):
                                    try:
                                        total_hits = int(
                                            line.split("Matched rules:", 1)[1].strip()
                                        )
                                    except ValueError:
                                        pass
                                    break
                    record["output_rule"] = rule_text
                    record["output_hit_sample_id"] = sample_id
                    record["output_total_hits"] = total_hits
            results.append(record)

        return results

    @staticmethod
    def _extract_final_text(messages: list) -> str:
        """Extract the last AI text from agent returned messages.

        For reasoning models (e.g. Doubao Seed reasoning series), the actual
        output may live in ``reasoning_content`` while ``content`` is empty.
        We fall back to ``reasoning_content`` (from response_metadata) when
        the normal content field carries no text.
        """
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts)

            # Fallback: reasoning models put the full answer in reasoning_content
            if not text.strip():
                reasoning_text = ""
                # Try response_metadata first (official LangChain path)
                resp_meta = getattr(msg, "response_metadata", None) or {}
                if isinstance(resp_meta, dict):
                    reasoning_text = resp_meta.get("reasoning_content", "") or ""
                # Also try the attribute directly, in case some wrapper exposes it
                if not reasoning_text:
                    reasoning_text = getattr(msg, "reasoning_content", "") or ""
                if reasoning_text.strip():
                    return reasoning_text
            elif text.strip():
                return text
        return ""

    @staticmethod
    def _parse_output(text: str, claim_type: str) -> dict[str, Any]:
        """Parse final answer and confidence from agent output."""
        text_clean = text.strip()

        # ── Extract final answer ──────────────────────────────────────
        final_answer = "Uncertain"

        if claim_type == "medical":
            # Multiple choice: look for A/B/C/D/E
            option_patterns = [
                r'(?:最终答案|answer)\s*(?:是|is|:)?\s*[：:]*\s*\[?([A-E])\]?\b',
                r'(?:答案|选项)\s*(?:是|为|:)?\s*[：:]*\s*\[?([A-E])\]?\b',
                r'\b\[?([A-E])\]?\s*(?:[\.\)]\s*|$)',
            ]
            for pattern in option_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).upper()
                    break
        else:
            # Yes/No task
            yes_patterns = [
                r'(?:最终答案|answer)\s*(?:是|is|:)?\s*[：:]*\s*\[?Yes\]?\b',
                r'\bAnswer\s*:\s*\[?Yes\]?\b',
                r'therefore,\s*the\s*final\s*answer\s*is\s*:\s*\[?Yes\]?',
            ]
            no_patterns = [
                r'(?:最终答案|answer)\s*(?:是|is|:)?\s*[：:]*\s*\[?No\]?\b',
                r'\bAnswer\s*:\s*\[?No\]?\b',
                r'therefore,\s*the\s*final\s*answer\s*is\s*:\s*\[?No\]?',
            ]
            for p in yes_patterns:
                if re.search(p, text_clean, re.IGNORECASE):
                    final_answer = "Yes"
                    break
            if final_answer == "Uncertain":
                for p in no_patterns:
                    if re.search(p, text_clean, re.IGNORECASE):
                        final_answer = "No"
                        break

        # ── Extract confidence ─────────────────────────────────────────
        confidence = 0.5
        conf_match = re.search(
            r'置信度[：:\s]*([0-9]*\.?[0-9]+)',
            text_clean,
        )
        if not conf_match:
            conf_match = re.search(
                r'confidence[：:\s]*([0-9]*\.?[0-9]+)',
                text_clean,
                re.IGNORECASE,
            )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        # ── Extract reasoning ───────────────────────────────────────
        reasoning = ""
        reason_match = re.search(
            r'<reasoning>(.*?)</reasoning>',
            text_clean,
            re.DOTALL,
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()
        else:
            reasoning = text_clean[:500]

        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    @staticmethod
    def _build_format_correction_prompt(text: str, claim_type: str) -> str:
        """Build a prompt to remind the LLM of the correct output format."""
        if claim_type == "medical":
            expected_answer = "A/B/C/D/E"
            example = "Therefore, the final answer is: A"
        else:
            expected_answer = "Yes/No"
            example = "Therefore, the final answer is: Yes"

        return f"""Your previous output was not in the correct format, so I couldn't parse your answer properly.

Your previous output:
{text}

Please follow the output format strictly:

1. Wrap your reasoning process with <reasoning> tags
2. Clearly state your final answer at the end

CORRECT FORMAT EXAMPLE:
<reasoning>
[Your three-stage reasoning process]
</reasoning>
{example}

Please re-answer with the correct format."""

    @staticmethod
    def _build_self_reflection_prompt(
        text: str,
        claim_type: str,
        current_answer: str,
        current_confidence: float,
    ) -> str:
        """Build a self-reflection prompt triggered by low confidence.

        When the model's first-pass confidence falls below the threshold,
        ask it to re-examine its reasoning along four bias/vigilance
        dimensions, then produce a revised answer with updated confidence.
        """
        if claim_type == "medical":
            example = "Therefore, the final answer is: A"
        else:
            example = "Therefore, the final answer is: Yes"

        dimensions_text = "\n".join(
            f"{i + 1}. **{dim}**"
            for i, dim in enumerate(SELF_REFLECTION_DIMENSIONS)
        )

        return f"""Your confidence ({current_confidence:.2f}) is below the threshold.
Please perform a structured self-reflection on your previous answer, then provide a revised judgment.

---

## Your previous answer
{text}

**Current answer:** {current_answer}
**Current confidence:** {current_confidence:.2f}

---

## Self-Reflection Instructions

Re-examine your reasoning carefully along the following four dimensions. For each one, honestly assess whether it may have biased your judgment, and explain how you would correct it.

{dimensions_text}

After reflecting on all four dimensions, re-examine the image and the claim, integrate your corrections, and produce your final revised answer.

You MAY change your answer or keep it the same, but you MUST justify your revised confidence based on the reflection.

---

## Output Format

Wrap your full reflection (all four dimensions + revised reasoning) with `<reasoning>` tags, then provide the final answer.

<reasoning>
[Your structured self-reflection across all four dimensions, followed by your revised reasoning]
</reasoning>
Confidence: [0.0-1.0]
{example}

Please provide your revised answer after self-reflection."""

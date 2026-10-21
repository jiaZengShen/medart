"""Single-agent v6 — Three-stage reasoning: Observe → Assess → Decide.

Based on v2 with ALL OBVISER mechanisms removed:
  - No OBVISER reflection recheck (no second LLM checking the output)
  - No laterality-consistency obviser (no image-flip blind re-judgment)

Kept from v2:
  - Correctly handle tool (content_blocks, result) return format
  - Support multi-round tool calls
  - Format-correction retry when the first answer parses to "Uncertain"
  - Debug tracker integration
  - No cache, clean and direct
"""

from __future__ import annotations

import json
import re
import time
import traceback
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool as langchain_tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.multi_agent.image_utils import build_image_data_url, is_image_too_small
from src.single_agent_v1.debug_tracker import DebugTracker
from src.single_agent_v1.prompts import (
    SINGLE_AGENT_SYSTEM_PROMPT,
    SINGLE_AGENT_HUMAN_TEMPLATE,
)


class SingleAgentV6:
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

        # Wrap tools to unify return format
        self._wrapped_tools = [self._wrap_tool(t) for t in tools]

        self._agent = create_agent(
            model=llm,
            tools=self._wrapped_tools,
            system_prompt=SINGLE_AGENT_SYSTEM_PROMPT,
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
            try:
                result = original_func(**kwargs)
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

        try:
            # ── Build user message ────────────────────────────────────
            claim = user_input.get("agent_prompt") or user_input.get("claim", "")
            context = user_input.get("context", "")
            claim_type = user_input.get("claim_type", "recognition")
            case_id = user_input.get("case_id", "unknown")
            image_paths = user_input.get("image_paths", [])

            # Debug: start snapshot & capture basics
            if self._debug_tracker is not None:
                self._debug_tracker.new_snapshot(case_id)
                self._debug_tracker.capture_basics(user_input)

            human_text = SINGLE_AGENT_HUMAN_TEMPLATE.substitute(
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

            # ── Format correction retry if parsed to Uncertain ───────
            if answer.get("final_answer") == "Uncertain":
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

            # Debug: capture main agent result
            if self._debug_tracker is not None:
                self._debug_tracker.capture_main_agent_result(
                    final_text=final_text,
                    parsed_answer=answer,
                    full_messages=all_messages,
                )

        except Exception as exc:
            traceback.print_exc()
            pipeline_status = "error"
            answer = {
                "final_answer": "Uncertain",
                "confidence": 0.0,
                "reasoning": f"Execution error: {exc}",
            }

        elapsed = time.time() - start_time

        # Build final output dict
        final_output = {
            "final_answer": answer.get("final_answer", "Uncertain"),
            "confidence": answer.get("confidence", 0.0),
            "reasoning": answer.get("reasoning", ""),
            "status": pipeline_status,
            "latency_s": round(elapsed, 2),
        }

        # Debug: capture final result
        if self._debug_tracker is not None:
            self._debug_tracker.capture_final_result(final_output)

        return final_output

    # ------------------------------------------------------------------
    # Helper functions for parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_final_text(messages: list) -> str:
        """Extract the last AI text from agent returned messages."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)
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

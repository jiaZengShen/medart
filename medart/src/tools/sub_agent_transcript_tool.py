import json
from typing import Any

from langchain.tools import tool as langchain_tool
from pydantic import BaseModel, Field

from src.multi_agent.shared_memory import SharedMemory


class AgentTranscriptInput(BaseModel):
    agent_id: str = Field(description="要读取的子 agent ID，例如 sub_1")


def build_sub_agent_transcript_tool(memory: SharedMemory):
    """构建给总结 agent 使用的工具：读取子 agent 完整 transcript。"""

    @langchain_tool(
        name_or_callable="get_sub_agent_transcript",
        description=(
            "读取一个成功子 agent 的完整执行记录。返回其推理消息、"
            "工具调用结果，以及工具返回中的图片 content blocks。"
        ),
        args_schema=AgentTranscriptInput,
        response_format="content_and_artifact",
    )
    def get_sub_agent_transcript(agent_id: str) -> tuple[Any, dict[str, Any]]:
        messages = memory.get_agent_messages(agent_id)
        if not messages:
            artifact = {
                "agent_id": agent_id,
                "found": False,
                "message": "未找到该子 agent 的执行记录",
            }
            return [
                {"type": "text", "text": f"未找到子 agent {agent_id} 的执行记录。"}
            ], artifact

        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"以下是子 agent {agent_id} 的完整执行记录。"}
        ]

        for idx, msg in enumerate(messages, start=1):
            msg_type = msg.get("type", "unknown")
            msg_content = msg.get("content", "")
            content_blocks.append(
                {"type": "text", "text": f"[{agent_id} message {idx} / {msg_type}]"}
            )
            if isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict) and block.get("type") in {"text", "image_url"}:
                        content_blocks.append(block)
                    else:
                        content_blocks.append(
                            {"type": "text", "text": json.dumps(block, ensure_ascii=False)}
                        )
            elif msg_content:
                content_blocks.append({"type": "text", "text": str(msg_content)})

        artifact = {
            "agent_id": agent_id,
            "found": True,
            "messages": messages,
        }
        return content_blocks, artifact

    return get_sub_agent_transcript

"""单智能体模块 —— 包含 V1 (三阶段+验证)、V2 (三阶段+OBVISER)、V3 (两阶段: 观察→判断)、V4 (组件验证)、V6 (三阶段, 无OBVISER)、V7 (问题拆解-合并)。"""

from src.single_agent_v1.agent import SingleAgentV1
from src.single_agent_v1.agent_v2 import SingleAgentV2
from src.single_agent_v1.agent_v3 import SingleAgentV3
from src.single_agent_v1.agent_v4 import SingleAgentV4
from src.single_agent_v1.agent_v6 import SingleAgentV6
from src.single_agent_v1.agent_v7 import SingleAgentV7
from src.single_agent_v1.agent_v8_2 import SingleAgentV8_2
from src.single_agent_v1.prompts import SINGLE_AGENT_SYSTEM_PROMPT, SINGLE_AGENT_HUMAN_TEMPLATE

__all__ = [
    "SingleAgentV1",
    "SingleAgentV2",
    "SingleAgentV3",
    "SingleAgentV4",
    "SingleAgentV6",
    "SingleAgentV7",
    "SingleAgentV8_2",
    "SINGLE_AGENT_SYSTEM_PROMPT",
    "SINGLE_AGENT_HUMAN_TEMPLATE",
]

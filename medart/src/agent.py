"""主单智能体文件，使用 RawChatOpenAI 构建 ReAct agent。"""

from langgraph.prebuilt import create_react_agent

from src.llms import RawChatOpenAI, PrintRawCallback
from src.prompts import AGENT_SYSTEM_PROMPT
from src.tools.hc_measurement import measure_fetal_head_circumference
from src.tools.biomedparse_client import biomedparse_segment
from src.tools.medsam_tool import medsam_segment





model = RawChatOpenAI(
    model="gpt-5.5",
    openai_api_key="*******",
    base_url="https://claude-api.cloud/v1",
    callbacks=[PrintRawCallback()],
)


model = RawChatOpenAI(
    model="doubao-seed-2.0-lite",
    openai_api_key="********",
    base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
    callbacks=[PrintRawCallback()],
)

from langchain.agents import create_agent

agent = create_react_agent(
    model=model,
    tools=[measure_fetal_head_circumference, biomedparse_segment, medsam_segment],
    prompt=AGENT_SYSTEM_PROMPT.format(tool_evidence="For measure-type claims, you may use the provided measurement tools to obtain verification evidence. For other claim types, ignore tools and answer based on visual analysis."),
)

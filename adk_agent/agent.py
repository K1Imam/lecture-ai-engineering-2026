from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app import tools
from app.config import MODEL

# Pydantic AI版と同じ指示文
INSTRUCTIONS = """\
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 確認問題や復習問題を求められたら make_quiz を使う。
- 学習記録を残すことが求められたら save_learning_log を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""


# Google ADK版エージェント
root_agent = Agent(
    name="aie2026_agent",
    model=MODEL,
    description="AIE2026の学習支援エージェント",
    instruction=INSTRUCTIONS,
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
        FunctionTool(
            func=tools.save_learning_log,
            require_confirmation=True,
        ),
    ],
)

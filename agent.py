from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import langchain as lk_langchain

from config import settings
from tools import ALL_TOOLS

load_dotenv(".env.local")

SYSTEM_PROMPT = (
    "You are a friendly voice-based HR assistant for AtliqAI.\n"
    "- Use search_hr_policies for any policy question.\n"
    "- Use check_leave_balance when the user gives an employee ID and asks about leave.\n"
    "- Use escalate_to_human if you cannot resolve the question with the tools above.\n"
    "Keep spoken answers very short and conversational — this is a voice call."
)


llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.0,
)

hr_graph = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


class HRAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are the AtliqAI HR voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=hr_graph),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.5,
                "max_delay": 6.0,
            },
        ),
    )

    await session.start(agent=HRAssistant(), room=ctx.room)

    await session.generate_reply(user_input="Greet the user as AtliqAI's HR assistant and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(server)
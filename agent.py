import asyncio

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import langchain as lk_langchain

from config import settings
from tools import ALL_TOOLS, call_state, sound

load_dotenv(".env.local")

SYSTEM_PROMPT = (
    "You are a friendly voice-based HR assistant for AtliqAI.\n"
    "- Use search_hr_policies for any policy question.\n"
    "- Use check_leave_balance when the user gives an employee ID and asks about leave.\n"
    "- Use escalate_to_human if you cannot resolve the question with the tools above.\n"
    "- Use end_call when the user says they are done — then say only a brief goodbye.\n"
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


def _is_tool_token(item: object) -> bool:
    """True for the (ToolMessage, metadata) pairs astream emits from the tools node."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class HRAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are the AtliqAI HR voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(hr_graph)),
        )


server = AgentServer()


# tell livekit to run this function for every new RTC session
@server.rtc_session()
async def entrypoint(ctx: agents.JobContext): #ctx contain everything related to the call, including the room and the user
    call_state.end_requested = False

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed", # dynamic
                "min_delay": 0.5,
                "max_delay": 3.0,
            },
        ),
    )

    await session.start(agent=HRAssistant(), room=ctx.room)

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.3),
        stream_timeout_ms=2000,
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    typing_handle: PlayHandle | None = None

    def start_typing() -> None:
        nonlocal typing_handle
        if typing_handle is None or typing_handle.done():
            typing_handle = background_audio.play(
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5, fade_out=0.4),
                loop=True,
            )

    def stop_typing() -> None:
        if typing_handle is not None:
            typing_handle.stop()

    sound.start_typing = start_typing
    sound.stop_typing = stop_typing

    async def hangup() -> None:
        await background_audio.aclose()
        await ctx.delete_room()
        ctx.shutdown(reason="user ended the call")

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
        # end_call raises the flag mid-turn, while the agent is still "thinking".
        # Returning to "listening" means the goodbye has finished playing out.
        if call_state.end_requested and ev.new_state == "listening":
            asyncio.create_task(hangup())

    await session.generate_reply(user_input="Greet the user as AtliqAI's HR assistant and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(server)
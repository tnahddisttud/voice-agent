import asyncio
from collections.abc import Callable

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings
from rag import rag

embedder = SentenceTransformer(settings.embedding_model)
qdrant = QdrantClient(path=settings.qdrant_path)

try:
    _count = qdrant.get_collection(settings.collection_name).points_count
    print(f"[OK] Reconnected to '{settings.collection_name}' - {_count} points")
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        f"Could not open Qdrant collection '{settings.collection_name}' "
        f"at '{settings.qdrant_path}'. Did you run `python ingest.py` first?"
    ) from exc


class SoundControl:
    """Lets a tool drive the background audio player without importing agent.py."""

    def __init__(self) -> None:
        self.start_typing: Callable[[], None] = lambda: None
        self.stop_typing: Callable[[], None] = lambda: None

sound = SoundControl()


@tool
async def search_hr_policies(query: str) -> str:
    """Search AtliqAI's HR policy documents for information relevant to the user's
    question. Use this whenever the user asks about leave, benefits, conduct,
    or any other HR policy topic."""
    sound.start_typing()
    try:
        answer, _ = await asyncio.to_thread(rag, query, 3, qdrant, embedder)
        return answer
    finally:
        sound.stop_typing()


# Mock HRMS data — in production this would be a real API call
MOCK_LEAVE_BALANCES = {
    "E1023": {"casual": 4, "sick": 6, "earned": 12},
    "E2044": {"casual": 1, "sick": 3, "earned": 8},
}


@tool
def check_leave_balance(employee_id: str) -> str:
    """Look up an employee's remaining leave balance by their employee ID
    (format: E followed by 4 digits, e.g. E1023). Use this when the user asks
    how much leave they have left."""
    balance = MOCK_LEAVE_BALANCES.get(employee_id.upper())
    if not balance:
        return f"No employee found with ID {employee_id}. Ask the user to double-check it."

    return (
        f"Casual: {balance['casual']} days, "
        f"Sick: {balance['sick']} days, "
        f"Earned: {balance['earned']} days."
    )


class CallState:
    """Shared between the end_call tool and the agent entrypoint.

    The tool can't hang up directly — it runs inside LangGraph and has no handle
    on the LiveKit session. It raises this flag instead, and agent.py tears the room down once the goodbye has finished playing.
    """

    def __init__(self) -> None:
        self.end_requested = False


call_state = CallState()


@tool
def end_call() -> str:
    """End the call and hang up. Use this only when the user clearly signals they
    are finished — "that's all", "goodbye", "nothing else, thanks". Do NOT use it
    if they ask to hold, pause, or want a human representative."""
    call_state.end_requested = True
    return "The call is ending. Say a short goodbye and nothing else."


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human HR representative when the question
    can't be answered from policy search or leave balance lookups — for example,
    disputes, sensitive personal situations, or repeated failed lookups."""
    ticket_id = "HR-" + str(abs(hash(reason)) % 10000)
    print(f"[MOCK ESCALATION] ticket={ticket_id} reason={reason!r}")
    return (
        f"I've created ticket {ticket_id} and a human HR representative will "
        f"follow up by email within one business day."
    )


ALL_TOOLS = [search_hr_policies, check_leave_balance, escalate_to_human, end_call]
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


@tool
def search_hr_policies(query: str) -> str:
    """Search AtliqAI's HR policy documents for information relevant to the user's
    question. Use this whenever the user asks about leave, benefits, conduct,
    or any other HR policy topic."""
    answer, _ = rag(query, top_k=3, client=qdrant, embedder=embedder)
    return answer


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


ALL_TOOLS = [search_hr_policies, check_leave_balance, escalate_to_human]
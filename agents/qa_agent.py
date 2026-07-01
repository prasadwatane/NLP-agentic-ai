"""
qa_agent.py — QAAgent (specialist).

Responsibility:
    Answer one executive question, grounded in the evidence the RetrievalAgent
    gathered and in the prior conversation. Its own LLM call. This is the
    conversational specialist used by the dashboard's "Live CEO Q&A" chat.

Input  : state["evidence_block"], state["question"], state["messages"] (history)
Output : state["answer"]
"""
import config as cfg
from agents import _common
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

RESPONDER_PROMPT = """Answer the executive's question about {company} using the evidence below
and the prior conversation. Be direct and decisive. Cite sources inline as [source].
If the evidence is thin, say so rather than inventing facts.

=== EVIDENCE ===
{evidence_block}
=== END EVIDENCE ===

Question: {question}"""


def run(state: _common.IntelState) -> dict:
    """LLM call: answer the question grounded in evidence + chat history."""
    sysmsg = _common.SYSTEM.format(company=state["company"], industry=state["industry"])
    user = RESPONDER_PROMPT.format(company=state["company"],
                                   evidence_block=state.get("evidence_block", ""),
                                   question=state["question"])
    history = state.get("messages", [])[:-1]   # prior turns precede the grounded question
    msgs = [SystemMessage(content=sysmsg)] + history + [HumanMessage(content=user)]
    answer = _common.invoke(msgs, temperature=state.get("temperature", cfg.TEMPERATURE),
                            json_mode=False)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}

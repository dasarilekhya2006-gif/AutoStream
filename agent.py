"""
AutoStream Conversational AI Agent
Social-to-Lead Agentic Workflow using LangGraph + Claude 3 Haiku
"""

import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ── Load Knowledge Base ──────────────────────────────────────────────────────

def load_knowledge_base(path: str = "knowledge_base.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

KB = load_knowledge_base()


# ── Mock Lead Capture Tool ────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ── RAG: Knowledge Retrieval ──────────────────────────────────────────────────

def retrieve_knowledge(query: str) -> str:
    """Simple keyword-based RAG over the local knowledge base."""
    query_lower = query.lower()
    results = []

    # Pricing info
    if any(word in query_lower for word in ["price", "pricing", "cost", "plan", "basic", "pro", "how much", "cheap", "afford"]):
        basic = KB["pricing"]["basic_plan"]
        pro = KB["pricing"]["pro_plan"]
        results.append(
            f"**Basic Plan** — ${basic['price_monthly']}/month: {', '.join(basic['features'])}\n"
            f"**Pro Plan** — ${pro['price_monthly']}/month: {', '.join(pro['features'])}"
        )

    # Refund policy
    if any(word in query_lower for word in ["refund", "money back", "cancel", "return"]):
        results.append(f"Refund Policy: {KB['policies']['refund_policy']}")

    # Support
    if any(word in query_lower for word in ["support", "help", "contact", "24/7"]):
        results.append(f"Support Policy: {KB['policies']['support']}")

    # FAQ matching
    for faq in KB["faq"]:
        if any(word in query_lower for word in faq["question"].lower().split()):
            results.append(f"Q: {faq['question']}\nA: {faq['answer']}")

    if not results:
        results.append(
            f"{KB['company']} is a SaaS platform offering automated video editing for content creators. "
            "Ask about pricing, features, or policies for more details."
        )

    return "\n\n".join(results)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history
    intent: Optional[str]                      # detected intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    awaiting: Optional[str]                    # which field we are collecting next


# ── LLM ──────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_retries=1,
            timeout=20,
        )
    except Exception as e:
        print(f"\n[warn] LLM init failed: {e}\n")
        llm = None


def heuristic_intent(text: str) -> str:
    """Offline fallback intent classifier when LLM/API is unavailable."""
    t = text.lower()
    if any(w in t for w in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "greeting"
    if any(w in t for w in ["buy", "start", "signup", "sign up", "try pro", "subscribe", "get started"]):
        return "high_intent"
    return "product_inquiry"


def safe_llm_content(messages: list, fallback_text: str) -> str:
    """Call LLM safely and return a usable text response even on API failures."""
    if llm is None:
        return fallback_text
    try:
        response = llm.invoke(messages)
        return str(response.content)
    except KeyboardInterrupt:
        return "Request interrupted. Please try again."
    except Exception as e:
        print(f"\n[warn] LLM request failed: {e}\n")
        return fallback_text

# ── Intent Detection ──────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing platform.
Classify the user's latest message into exactly one of:
  - greeting        (hello, hi, hey, general opener with no product question)
  - product_inquiry (questions about features, pricing, plans, support, refunds, etc.)
  - high_intent     (user is ready to sign up / try / buy / start a plan)

Reply with ONLY the label, nothing else."""

def detect_intent(state: AgentState) -> AgentState:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    if not GOOGLE_API_KEY:
        intent = heuristic_intent(last_human)
        return {**state, "intent": intent}

    content = safe_llm_content([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_human),
    ], fallback_text=heuristic_intent(last_human))
    intent = str(content).strip().lower()
    if intent not in {"greeting", "product_inquiry", "high_intent"}:
        intent = "product_inquiry"
    return {**state, "intent": intent}


# ── Router ────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    if state.get("lead_captured"):
        return "done"
    # If we are mid-collection, stay in collection loop
    if state.get("awaiting"):
        return "collect_lead"
    intent = state.get("intent", "product_inquiry")
    if intent == "greeting":
        return "greet"
    elif intent == "high_intent":
        return "collect_lead"
    else:
        return "rag_response"


# ── Node: Greet ───────────────────────────────────────────────────────────────

GREET_SYSTEM = """You are a friendly sales assistant for AutoStream, an automated video editing SaaS for content creators.
Greet the user warmly and briefly introduce what AutoStream does.
Invite them to ask about pricing, features, or to get started.
Keep it to 2-3 sentences."""

def greet_node(state: AgentState) -> AgentState:
    fallback = (
        "Hey! Welcome to AutoStream. We help creators automate video editing at scale. "
        "Ask about pricing, features, or tell me if you want to get started."
    )
    content = safe_llm_content([SystemMessage(content=GREET_SYSTEM)] + state["messages"], fallback)
    return {**state, "messages": state["messages"] + [AIMessage(content=content)]}


# ── Node: RAG Response ────────────────────────────────────────────────────────

def rag_response_node(state: AgentState) -> AgentState:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    kb_context = retrieve_knowledge(last_human)

    system = f"""You are a helpful sales assistant for AutoStream, a SaaS video editing platform.
Use ONLY the context below to answer the user's question accurately and concisely.
If the answer is not in the context, say you'll connect them with the team.

Context:
{kb_context}

After answering, gently ask if they'd like to get started or have more questions."""

    fallback = (
        f"{kb_context}\n\n"
        "If you'd like, I can also help you get started by collecting your name, email, and creator platform."
    )
    content = safe_llm_content([SystemMessage(content=system)] + state["messages"], fallback)
    return {**state, "messages": state["messages"] + [AIMessage(content=content)]}


# ── Node: Collect Lead ────────────────────────────────────────────────────────

def collect_lead_node(state: AgentState) -> AgentState:
    # Check if last message has the value we were awaiting
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    new_state = dict(state)

    awaiting = state.get("awaiting")

    # Persist collected values
    if awaiting == "name":
        new_state["lead_name"] = last_human
        new_state["awaiting"] = "email"
    elif awaiting == "email":
        new_state["lead_email"] = last_human
        new_state["awaiting"] = "platform"
    elif awaiting == "platform":
        new_state["lead_platform"] = last_human
        new_state["awaiting"] = None

    # Check if we now have everything
    if new_state.get("lead_name") and new_state.get("lead_email") and new_state.get("lead_platform"):
        result = mock_lead_capture(
            new_state["lead_name"],
            new_state["lead_email"],
            new_state["lead_platform"],
        )
        new_state["lead_captured"] = True
        reply = (
            f"🎉 You're all set! We've captured your details:\n"
            f"- **Name**: {new_state['lead_name']}\n"
            f"- **Email**: {new_state['lead_email']}\n"
            f"- **Platform**: {new_state['lead_platform']}\n\n"
            f"Our team will reach out shortly to help you get started on AutoStream's Pro plan. "
            f"Welcome aboard! 🚀"
        )
        new_state["messages"] = state["messages"] + [AIMessage(content=reply)]
        return new_state

    # Determine what to ask next
    if not new_state.get("lead_name") and not awaiting:
        new_state["awaiting"] = "name"
        reply = (
            "Awesome! It sounds like you're ready to get started. "
            "I'll just need a few quick details.\n\nWhat's your **name**?"
        )
    elif new_state.get("awaiting") == "email":
        reply = f"Thanks, {new_state.get('lead_name', '')}! What's your **email address**?"
    elif new_state.get("awaiting") == "platform":
        reply = "Great! Which **creator platform** do you primarily use? (e.g., YouTube, Instagram, TikTok)"
    else:
        reply = "Let me get a few details to get you started. What's your **name**?"
        new_state["awaiting"] = "name"

    new_state["messages"] = state["messages"] + [AIMessage(content=reply)]
    return new_state


# ── Node: Done ────────────────────────────────────────────────────────────────

def done_node(state: AgentState) -> AgentState:
    reply = "Is there anything else I can help you with? 😊"
    return {**state, "messages": state["messages"] + [AIMessage(content=reply)]}


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("greet", greet_node)
    graph.add_node("rag_response", rag_response_node)
    graph.add_node("collect_lead", collect_lead_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges("detect_intent", route, {
        "greet": "greet",
        "rag_response": "rag_response",
        "collect_lead": "collect_lead",
        "done": "done",
    })

    graph.add_edge("greet", END)
    graph.add_edge("rag_response", END)
    graph.add_edge("collect_lead", END)
    graph.add_edge("done", END)

    return graph.compile()


# ── CLI Chat Loop ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AutoStream AI Sales Agent")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    app = build_graph()

    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "awaiting": None,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        try:
            state = app.invoke(state)
        except KeyboardInterrupt:
            print("\nAgent: Request interrupted. You can continue chatting or type 'exit'.")
            continue

        # Print last AI message
        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        print(f"\nAgent: {last_ai}")

        if state.get("lead_captured"):
            more = input("\nYou: ").strip()
            if more:
                print("\nAgent: Thanks for chatting! Feel free to reach out anytime. 🚀")
            break


if __name__ == "__main__":
    main()

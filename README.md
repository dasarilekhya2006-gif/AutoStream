# AutoStream 

A conversational AI sales agent built with **LangGraph** and **Google Gemini**, designed to handle social-to-lead workflows for AutoStream — a SaaS video editing platform for content creators.

---

## Features

- **Intent Detection** — Classifies each user message as a greeting, product inquiry, or high-intent (ready to buy)
- **RAG (Retrieval-Augmented Generation)** — Answers questions using a local knowledge base covering pricing, policies, and FAQs
- **Lead Capture Flow** — Automatically collects name, email, and creator platform when a user signals purchase intent
- **Offline Fallback** — Heuristic intent classifier and pre-written responses work even without an active API connection
- **LangGraph State Machine** — Clean, modular graph with nodes for greeting, RAG response, lead collection, and completion

---

## Project Structure

```
.
├── agent.py              # Main agent logic (LangGraph graph + nodes)
├── knowledge_base.json   # Product info, pricing, FAQs, and policies
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (API keys)
```

---

## Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key (for Gemini)

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd autostream-agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**

   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   > ⚠️ Never commit your `.env` file to version control. Add it to `.gitignore`.

---

## Running the Agent

```bash
python agent.py
```

You'll see an interactive CLI prompt:

```
============================================================
  AutoStream AI Sales Agent
  Type 'exit' to quit.
============================================================

You: 
```

Type any message to start chatting. Type `exit` or `quit` to stop.

---

## Example Conversation

```
You: Hey there!
Agent: Hey! Welcome to AutoStream. We help creators automate video editing at scale.
       Ask about pricing, features, or tell me if you want to get started.

You: What's included in the Pro plan?
Agent: The Pro Plan is $79/month and includes unlimited videos, 4K resolution,
       AI captions, advanced editing tools, and 24/7 priority support. ...

You: I want to sign up!
Agent: Awesome! It sounds like you're ready to get started. What's your name?

You: Alex
Agent: Thanks, Alex! What's your email address?

You: alex@example.com
Agent: Great! Which creator platform do you primarily use? (e.g., YouTube, Instagram, TikTok)

You: YouTube
Agent: 🎉 You're all set! We've captured your details:
       - Name: Alex
       - Email: alex@example.com
       - Platform: YouTube
       Our team will reach out shortly. Welcome aboard! 🚀
```

---

## Agent Architecture

```
User Message
     │
     ▼
[detect_intent]
     │
     ├── greeting      → [greet]         → END
     ├── product_inquiry → [rag_response] → END
     ├── high_intent   → [collect_lead]  → END
     └── lead_captured → [done]          → END
```

### Nodes

| Node | Description |
|---|---|
| `detect_intent` | Classifies user intent via Gemini or heuristic fallback |
| `greet` | Sends a warm welcome message introducing AutoStream |
| `rag_response` | Retrieves relevant info from `knowledge_base.json` and responds |
| `collect_lead` | Sequentially collects name → email → platform, then captures the lead |
| `done` | Asks if the user needs anything else after lead capture |

---

## Knowledge Base

The `knowledge_base.json` file contains:

- **Pricing** — Basic ($29/mo) and Pro ($79/mo) plan details
- **Policies** — Refund, cancellation, support, and free trial policies
- **FAQs** — Common questions about platforms, video formats, AI captions, and plan changes

To customize the agent for a different product, update this file — no code changes needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | State machine / graph orchestration |
| `langchain` | LLM abstraction and message types |
| `langchain-google-genai` | Gemini model integration |
| `langchain-core` | Core message and runnable primitives |
| `python-dotenv` | Load environment variables from `.env` |

---

## Notes

- The agent runs fully offline (with heuristic fallbacks) if the Gemini API is unavailable.
- The lead capture in `mock_lead_capture()` is a stub — replace it with your CRM integration (HubSpot, Salesforce, etc.).
- Conversation state is held in memory for the duration of one CLI session and is not persisted.

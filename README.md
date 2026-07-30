# AtliqAI HR Voice Assistant 🎙️🤖

A real-time, voice-enabled AI assistant for HR policy inquiries and employee services, powered by **LiveKit Agents**, **LangChain**, **Groq (Llama 3.3 70B)**, **Docling**, and **Qdrant Vector DB**.

---

## 🌟 Overview

The **AtliqAI HR Voice Assistant** enables employees to speak directly with an AI HR representative over a live audio stream. The assistant answers company policy questions using Retrieval-Augmented Generation (RAG) over HR policy documents, checks employee leave balances, and escalates complex issues to human HR representatives when necessary.

---

## ✨ Features

- **🎙️ Real-Time Voice Interaction**: Seamless speech-to-text (Deepgram Nova-3) and text-to-speech (Inworld Ashley) streaming via LiveKit Agents.
- **📄 Hierarchical Document RAG**: Automatic parsing and chunking of HR policy PDFs using Docling's `HierarchicalChunker`, preserving document headings and structure.
- **⚡ Fast Vector Retrieval**: Local vector store indexing with Qdrant and `sentence-transformers` (`all-MiniLM-L6-v2`).
- **🧠 LangChain Agent Tool Calling**:
  - `search_hr_policies`: Semantic retrieval across company HR policies.
  - `check_leave_balance`: Look up remaining casual, sick, and earned leave (Mock HRMS integration).
  - `escalate_to_human`: Automatically create HR support tickets for unresolved or sensitive inquiries.
- **⚙️ Environment & Config Management**: Centralized Pydantic settings loading from `.env`.

---

## 🏗️ Architecture

```text
                  +-----------------------------------+
                  |      LiveKit RTC Audio Stream     |
                  +-----------------+-----------------+
                                    |
            +-----------------------v-----------------------+
            |                  Agent Session                |
            |   STT (Deepgram) ------------> TTS (Inworld)  |
            +-----------------------+-----------------------+
                                    |
                                    v
                         +--------------------+
                         |  LangChain Agent   |
                         |  (Groq Llama 3.3)  |
                         +---------+----------+
                                   |
           +-----------------------+-----------------------+
           |                       |                       |
           v                       v                       v
 [ search_hr_policies ]  [ check_leave_balance ]  [ escalate_to_human ]
           |                       |                       |
           v                       v                       v
     Qdrant RAG              Mock HR Database       Ticket System
```

---

## 📁 Repository Structure

- [`agent.py`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/agent.py) - LiveKit Agent worker entrypoint and LLM adapter setup.
- [`ingest.py`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/ingest.py) - Document downloader, Docling parser, vector embedding generator, and Qdrant ingestion script.
- [`rag.py`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/rag.py) - RAG retrieval logic and standalone CLI query runner.
- [`tools.py`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/tools.py) - Definitions for tools made available to the agent.
- [`config.py`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/config.py) - Pydantic settings loader reading from environment variables.
- [`.env.example`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/.env.example) - Template for required and optional environment variables.
- [`pyproject.toml`](file:///d:/projects/Courses/Bootcamps/AI-ENG/voice-agent/pyproject.toml) - Project dependencies and metadata.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `>= 3.12`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- **API Credentials**:
  - LiveKit Cloud credentials (URL, API Key, API Secret)
  - Groq API Key

---

### 1. Installation

Clone the repository and install the dependencies:

```bash
# Install dependencies with uv
uv sync

# Or using pip
pip install -e .
```

---

### 2. Environment Configuration

Copy the example environment configuration file to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

GROQ_API_KEY=your_groq_api_key
```

---

### 3. Ingesting HR Documents

Before running the voice agent, process and index the HR policy document into the local Qdrant database:

```bash
python ingest.py
```

*This downloads the HR policy PDF, parses document structure using Docling, creates embeddings, and stores them in Qdrant.*

---

### 4. Testing RAG directly (CLI Mode)

You can test the RAG pipeline directly from the command line:

```bash
python rag.py "How many casual leaves am I entitled to?"
```

---

### 5. Running the Voice Agent

Start the LiveKit agent worker in development mode:

```bash
python agent.py dev
```

Connect using the [LiveKit Agents Sandbox](https://agents-playground.livekit.io/) or your custom frontend client to initiate a voice conversation with the agent.

---

## 🧪 Demo Employee Data (Leave Balance Lookup)

When testing the `check_leave_balance` tool during a voice call, try using these sample employee IDs:

| Employee ID | Casual Leaves | Sick Leaves | Earned Leaves |
| :--- | :--- | :--- | :--- |
| `E1023` | 4 days | 6 days | 12 days |
| `E2044` | 1 day | 3 days | 8 days |

---

## 🛡️ License

This project is open-source and intended for educational and bootcamp demonstrations.

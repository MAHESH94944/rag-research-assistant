# 🧠 RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)
![Vertex AI](https://img.shields.io/badge/VertexAI-Gemini-red)

A production-grade **Multi-Agent RAG (Retrieval-Augmented Generation) Research System** built using **LangChain**, **LangGraph**, **FastAPI**, **FAISS**, and **Google Vertex AI Gemini**.

This system autonomously researches any topic from the internet, performs semantic retrieval using vector databases, generates professional research reports, and evaluates its own outputs using AI-based feedback.

---

# 🚀 Features

- 🔍 Autonomous web research
- 🌐 Multi-source web scraping
- 🧠 RAG pipeline using embeddings + vector search
- 🤖 Multi-agent architecture with LangGraph
- ⚡ Async scraping for performance
- 📝 Professional report generation
- 📚 Source citation tracking
- ⭐ AI-based evaluation & feedback
- 💾 Persistent memory using ChromaDB
- 📡 FastAPI backend for API access
- 🔄 Streaming support
- 🧪 Modular testing support

---

# 🏗️ System Architecture

```text
User Query
    ↓
Planner Agent
    ↓
Search Agent (Tavily)
    ↓
Async Web Scraper
    ↓
Text Cleaning + Chunking
    ↓
Embeddings Generation
    ↓
FAISS Vector Database
    ↓
Retriever
    ↓
Writer Agent (Gemini)
    ↓
Critic Agent
    ↓
Final Research Report
```

---

# 🤖 Multi-Agent Workflow

| Agent | Responsibility |
|---|---|
| Planner Agent | Research planning and workflow decisions |
| Search Agent | Searches the web using Tavily |
| Scraper Agent | Extracts content from URLs |
| Chunker | Splits large text intelligently |
| Retriever | Retrieves semantically relevant chunks |
| Writer Agent | Generates final report |
| Critic Agent | Evaluates quality and hallucinations |
| Memory System | Stores persistent research history |

---

# 🧠 RAG Pipeline

This project implements a complete Retrieval-Augmented Generation pipeline:

## Steps

1. Search relevant web pages
2. Scrape content
3. Clean extracted text
4. Split into chunks
5. Generate embeddings
6. Store in FAISS vector DB
7. Retrieve relevant chunks
8. Generate grounded report
9. Evaluate report quality

---

# 🛠️ Tech Stack

## AI / LLM Frameworks

- LangChain
- LangGraph
- Vertex AI Gemini
- Mistral AI

## RAG & Vector Databases

- FAISS
- ChromaDB
- sentence-transformers

## Backend

- FastAPI
- Uvicorn

## Web Scraping

- BeautifulSoup4
- aiohttp
- requests
- Selenium

## Utilities

- Pydantic
- Loguru
- Tenacity
- Rich

---

# 📂 Project Structure

```text
RAG_Project/
│
├── main.py                     # FastAPI backend
├── pipeline.py                 # Sequential pipeline
├── langgraph_workflow.py       # LangGraph orchestration
│
├── agents.py                   # Writer & critic agents
├── planner_agent.py            # Planner logic
├── tools.py                    # Search + scraping tools
├── async_scraper.py            # Concurrent scraping
│
├── rag.py                      # RAG implementation
├── memory_system.py            # ChromaDB memory
├── evaluation_system.py        # Evaluation pipeline
│
├── streaming_output.py         # Streaming responses
│
├── requirements.txt
├── requirements_api.txt
├── .env
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-research-assistant.git

cd rag-research-assistant
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
# =====================================================
# GOOGLE VERTEX AI
# =====================================================

GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL_NAME=gemini-2.0-flash

# =====================================================
# GOOGLE CREDENTIALS
# =====================================================

GOOGLE_APPLICATION_CREDENTIALS=dev-google-credentials.json

# =====================================================
# SEARCH API
# =====================================================

TAVILY_API_KEY=your_tavily_api_key

# =====================================================
# OPTIONAL
# =====================================================

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
MAX_SEARCH_RESULTS=10
```

---

# ▶️ Running the Project

## Run Full Pipeline

```bash
python pipeline.py
```

---

## Run FastAPI Server

```bash
python main.py
```

OR

```bash
uvicorn main:app --reload
```

---

# 📡 API Endpoints

---

## POST `/research`

Researches any topic and generates a professional report.

### Request

```json
{
  "topic": "How NVIDIA became valuable",
  "use_advanced": false,
  "return_feedback": true
}
```

---

### Response

```json
{
  "request_id": "req_xxxxx",
  "topic": "How NVIDIA became valuable",
  "report": "Generated research report...",
  "feedback": "AI evaluation feedback...",
  "sources": [],
  "metadata": {}
}
```

---

## GET `/health`

Health check endpoint.

---

# 🧪 Testing Components

## Test RAG

```bash
python rag.py
```

---

## Test Memory System

```bash
python memory_system.py
```

---

## Test Planner Agent

```bash
python planner_agent.py
```

---

## Test Evaluation System

```bash
python evaluation_system.py
```

---

## Test Streaming

```bash
python streaming_output.py
```

---

# 📊 Example Output

```markdown
## Introduction

NVIDIA's rise to becoming one of the most valuable companies
in the world is driven by strategic GPU innovation,
AI infrastructure dominance, and ecosystem expansion.

## Key Findings

### GPU Architecture Advantage

NVIDIA's GPUs became the backbone of AI computation.

### CUDA Ecosystem

CUDA created strong developer lock-in and ecosystem growth.

### AI Revolution

Massive AI demand accelerated NVIDIA's market valuation.

## Conclusion

NVIDIA successfully positioned itself as the leading AI
hardware and infrastructure provider.
```

---

# ⚡ Key Engineering Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Multi-Agent Systems
- LangGraph Workflows
- Vector Databases
- Semantic Search
- Embeddings
- Async Processing
- Streaming LLM Responses
- AI Evaluation Pipelines
- Persistent Memory Systems
- FastAPI Backend Development

---

# 📈 Performance

| Operation | Approx Time |
|---|---|
| Web Search | 3-5 sec |
| Scraping | 5-10 sec |
| Vector Embedding | 2-3 sec |
| Report Generation | 10-20 sec |
| Total Pipeline | 25-40 sec |

---

# 🎯 Why This Project Matters

This project demonstrates real-world AI engineering concepts used in production GenAI systems:

- Enterprise-grade RAG pipelines
- Autonomous research agents
- AI orchestration frameworks
- Scalable backend architecture
- LLM evaluation systems
- Retrieval systems with semantic understanding

---

# 🔒 Security Notes

- API keys are stored in `.env`
- Credentials are excluded using `.gitignore`
- Never push service account JSON files to GitHub

---

# 📄 .gitignore

Create a `.gitignore` file:

```gitignore
# Secrets
.env
*.json
*.pem
*.key

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual Environment
.venv/
venv/

# Vector Stores
vector_store/
research_memory/
faiss_index/

# Generated Files
research_*.txt

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

# 📚 Future Improvements

- Hybrid search (BM25 + Vector Search)
- Multi-modal RAG
- PDF and DOCX ingestion
- Redis caching
- Advanced evaluation metrics
- Real-time streaming frontend
- Kubernetes deployment
- Authentication & rate limiting

---

# 👨‍💻 Author

**Mahesh Jadhao**

AI Engineering • RAG Systems • LangGraph • FastAPI • GenAI

---

# ⭐ Support

If you found this project useful, consider giving it a star ⭐

---

# 📜 License

MIT License

Feel free to use this project for learning and portfolio purposes.
# Legal Expert Assistant - RAG System for Indian Law

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA%203.2-orange.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-red.svg)

*An intelligent legal assistant powered by RAG (Retrieval-Augmented Generation) for Indian Law*

</div>

---

## 📋 Overview

**Legal Expert Assistant** is a production-ready RAG (Retrieval-Augmented Generation) system that provides accurate, context-aware answers to legal questions based on Indian law. The system leverages vector embeddings and semantic search to retrieve relevant legal provisions from a curated knowledge base, then uses a large language model to generate comprehensive, well-cited responses.

### 🎯 Key Features

- **Multi-Source Legal Knowledge Base**: Includes the Indian Penal Code (IPC), Bharatiya Nyaya Sanhita 2023 (BNS), and the Constitution of India
- **Semantic Search**: Uses HuggingFace embeddings (all-MiniLM-L6-v2) with ChromaDB for efficient vector similarity search
- **Local LLM Integration**: Runs entirely locally using Ollama with LLaMA 3.2 - no API costs or data privacy concerns
- **Interactive Web Interface**: Clean, user-friendly Gradio interface with real-time context display
- **Comprehensive Evaluation Framework**: Built-in evaluation dashboard with MRR, nDCG, and LLM-as-a-Judge metrics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Gradio Web App - app.py)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Question  │───▶│  Retriever  │───▶│   LLM (LLaMA 3.2)   │  │
│  │   Input     │    │  (Top-K)    │    │   Response Gen      │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│                    ┌─────────────┐                               │
│                    │  ChromaDB   │                               │
│                    │Vector Store │                               │
│                    └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Knowledge Base                              │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │   IPC    │    │      BNS        │    │  Constitution    │   │
│  │(Criminal)│    │(New Penal Code) │    │    of India      │   │
│  └──────────┘    └─────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
legal-expert-assistant/
├── app.py                    # Main Gradio chat application
├── evaluator.py              # RAG evaluation dashboard
├── implementation/
│   ├── answer.py             # RAG pipeline (retrieval + generation)
│   └── ingest.py             # Document ingestion and embedding
├── evaluation/
│   ├── eval.py               # Evaluation metrics (MRR, nDCG, LLM-judge)
│   ├── test.py               # Test case data model
│   └── tests.jsonl           # Test dataset with reference answers
├── knowledge-base/
│   ├── ipc.txt               # Indian Penal Code
│   ├── bns.txt               # Bharatiya Nyaya Sanhita 2023
│   └── coi.txt               # Constitution of India
├── vector_db/                # ChromaDB persistent storage
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- LLaMA 3.2 model pulled: `ollama pull llama3.2:latest`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-expert-assistant.git
   cd legal-expert-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

5. **Ingest the knowledge base** (first-time setup)
   ```bash
   python implementation/ingest.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

The web interface will open automatically at `http://localhost:7860`

---

## 💡 Usage

### Chat Interface

Ask any question about Indian law:

- *"What is Section 302 of the Indian Penal Code?"*
- *"What are the Fundamental Rights in the Constitution?"*
- *"How does the BNS 2023 define criminal conspiracy?"*
- *"What is the Right to Private Defence under IPC?"*

The system will:
1. Retrieve relevant legal provisions from the knowledge base
2. Display the source documents in the context panel
3. Generate a comprehensive answer with proper citations

### Evaluation Dashboard

Run the evaluation dashboard to assess system performance:

```bash
python evaluator.py
```

This provides:
- **Retrieval Metrics**: MRR, nDCG, Keyword Coverage
- **Answer Quality**: Accuracy, Completeness, Relevance (LLM-as-Judge)
- **Category Breakdown**: Performance by question type

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **MRR** | Mean Reciprocal Rank - how quickly relevant docs are retrieved | ≥ 0.9 |
| **nDCG** | Normalized Discounted Cumulative Gain | ≥ 0.9 |
| **Coverage** | Percentage of keywords found in retrieved context | ≥ 90% |
| **Accuracy** | Factual correctness vs reference answer | ≥ 4.5/5 |
| **Completeness** | How thoroughly the answer addresses the question | ≥ 4.5/5 |
| **Relevance** | How directly the answer addresses the query | ≥ 4.5/5 |

---

## 🔧 Configuration

### Model Configuration

In `implementation/answer.py`:

```python
MODEL = "llama3.2:latest"           # Ollama model
RETRIEVAL_K = 10                     # Number of chunks to retrieve
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### Chunking Configuration

In `implementation/ingest.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=200    # Overlap between chunks
)
```

---

## 🛠️ Technical Details

### RAG Pipeline

1. **Document Ingestion**: Legal documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`
2. **Embedding Generation**: Chunks are embedded using `all-MiniLM-L6-v2` (384 dimensions)
3. **Vector Storage**: Embeddings stored in ChromaDB for persistent, efficient retrieval
4. **Semantic Search**: User queries are embedded and matched against the vector store
5. **Context Injection**: Top-K relevant chunks are injected into the LLM prompt
6. **Response Generation**: LLaMA 3.2 generates a comprehensive, cited response

### LLM-as-Judge Evaluation

The system uses LLM-as-Judge methodology with structured outputs (Pydantic models) to evaluate:
- Answer accuracy against reference answers
- Completeness of legal information
- Relevance to the specific question asked

---

## 📚 Knowledge Base

| Document | Description | Coverage |
|----------|-------------|----------|
| **IPC** | Indian Penal Code | Criminal offences, punishments, defences |
| **BNS 2023** | Bharatiya Nyaya Sanhita | Modern replacement for IPC (enacted Dec 2023) |
| **Constitution** | Constitution of India | Fundamental Rights, Duties, Government structure |

---

## 🧪 Running Tests

```bash
# Run evaluation on a specific test case
python evaluation/eval.py 0

# Run full evaluation suite (via dashboard)
python evaluator.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Gradio](https://gradio.app/) for the web interface
- [HuggingFace](https://huggingface.co/) for embedding models

---

<div align="center">

**Built with ❤️ for the legal community**

*Disclaimer: This tool is for informational purposes only and does not constitute legal advice.*

</div>

# 📁 FileChat

**FileChat** is a modern, AI-powered document Q&A system. Upload PDFs, ask questions, and get instant, context-aware answers using advanced language models and semantic search.

---

## 🚀 Features

- **Multi-PDF Support**: Upload and query multiple documents.
- **Semantic Search**: Fast, accurate retrieval using FAISS vector store.
- **Chunked Processing**: Smart text chunking for better context.
- **LLM Integration**: Uses state-of-the-art language models for answers.
- **Clean UI & Logging**: Simple interface, robust logging for debugging.

---

## ⚡ Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/FileChat.git
   cd FileChat



graph TB
    %% User Interface Layer
    subgraph "🎨 User Interface"
        UI[🖥️ app.py (Main App)]
    end

    %% Database Layer
    subgraph "🗄️ Database Layer (SQLite)"
        DB[(🗄️ chatbot.db)]
        DBModule[🗃️ src/db/database.py]
    end

    %% Data & Storage
    subgraph "📂 Data Storage"
        PDFs[📄 data/*.pdf]
        FAISSIndex[🔍 data/faiss_index/]
        EmbeddingsCache[🗂️ data/embeddings_cache.pkl]
        Logs[📝 data/logs/app.log]
    end

    %% Core Pipeline
    subgraph "⚙️ Core Pipeline"
        PDFLoader[📖 src/loader/pdf_loader.py]
        TextSplitter[✂️ src/splitter/semantic_chunker.py]
        TextCleaner[🧹 src/utils/text_cleaner.py]
        VectorStore[🔍 src/vector_store/faiss_store.py]
        PipelineCore[🏗️ src/pipeline/core.py]
    end

    %% LLM & Processing
    subgraph "🧠 AI Processing"
        LLM[🤖 src/llm/llm.py]
    end

    %% Workflow & State
    subgraph "🔄 Workflow & State"
        Workflow[🔗 src/graph/workflow.py]
        Nodes[🧩 src/graph/nodes.py]
        State[🗺️ src/graph/state.py]
    end

    %% Notebooks
    subgraph "📒 Notebooks"
        Notebook[📓 notebooks/main.ipynb]
    end

    %% Data Flow
    UI --> PDFLoader
    PDFLoader --> PDFs
    PDFLoader --> TextSplitter
    TextSplitter --> TextCleaner
    TextCleaner --> VectorStore
    VectorStore --> FAISSIndex
    VectorStore --> EmbeddingsCache
    VectorStore --> PipelineCore
    PipelineCore --> LLM
    LLM --> Workflow
    Workflow --> Nodes
    Workflow --> State
    UI --> DBModule
    DBModule --> DB
    UI --> Logs
    Notebook --> PipelineCore

    %% Styling
    classDef uiClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px

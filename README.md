# FileChat

> **Intelligent File Chat - Powered by AI**

Transform your documents into conversational knowledge. Ask questions, get instant answers, and unlock insights from your PDFs, documents, and files using advanced AI technology.

## ✨ Features

- **🤖 AI-Powered Chat** - Natural language conversations with your documents
- **📄 Multi-Format Support** - PDF, Word, and text file compatibility
- **🔍 Smart Retrieval** - Advanced semantic search and BM25 retrieval
- **💾 Persistent Memory** - Embeddings cached for lightning-fast responses
- **🌐 Web Interface** - Clean, intuitive Streamlit-based UI
- **⚡ Real-time Processing** - Instant answers without regenerating embeddings

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys (see Configuration)

### Installation
```bash
git clone https://github.com/yourusername/DocuChat-AI.git
cd DocuChat-AI
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file in the root directory
2. Add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Run the Application
```bash
streamlit run app.py
```

## 🏗️ Architecture

```
src/
├── config.py      # Configuration management
├── graph.py       # LangGraph workflow
├── LLM.py        # Language model integration
├── nodes.py      # Processing nodes
├── state.py      # State management
├── store.py      # Document storage & retrieval
└── utils.py      # Utility functions
```

## 🔧 How It Works

1. **Upload** - Drag & drop your document
2. **Process** - AI generates embeddings (once per document)
3. **Chat** - Ask questions in natural language
4. **Retrieve** - Smart search finds relevant content
5. **Answer** - AI generates contextual responses

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Framework**: LangGraph, LangChain
- **Embeddings**: HuggingFace Transformers
- **Vector Store**: FAISS
- **Search**: BM25 + Semantic Retrieval
- **Language Models**: OpenAI, Mistral AI

## 📁 Project Structure

```
Chat_With_PDF/
├── app.py                 # Main Streamlit application
├── src/                   # Core source code
├── data/                  # Document storage
├── notebooks/             # Development notebooks
└── requirements.txt       # Dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Streamlit](https://streamlit.io/)
- AI capabilities from [OpenAI](https://openai.com/) and [Mistral AI](https://mistral.ai/)

---

**Made with ❤️ for intelligent document processing** 
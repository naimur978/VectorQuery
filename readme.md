<h1 align="center"> InfoGPT </h1>

<div align="center">

  
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Llama2](https://img.shields.io/badge/Llama2-purple?style=for-the-badge) ![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)




</div>


## Purpose

**InfoGPT** is an intelligent document Q&A system that empowers you to have conversations with your own documents using cutting-edge Large Language Models (LLMs) and vector databases. With an intuitive Streamlit-based web interface, InfoGPT enables you to upload and index various document types (PDF, TXT, CSV, DOCX, etc.), then query them using natural language to receive accurate, contextually-aware answers.

The system leverages **Retrieval Augmented Generation (RAG)** technology to combine semantic document search with powerful AI models, ensuring responses are grounded in your actual document content. This approach eliminates hallucinations and provides reliable, source-backed answers.

**Supported AI Models:**
- 🧠 **Google Gemini 2.5 Flash** - Latest Google AI model with excellent performance
- 🤖 **OpenAI GPT Models** - Industry-standard ChatGPT models  
- 💻 **Local LLama Models** - Privacy-focused local inference (GGUF format)

InfoGPT is perfect for researchers, students, professionals, legal teams, and anyone who needs to quickly extract insights from their document collections while maintaining full control over their data.

---

Harness the capabilities of state-of-the-art Large Language Models combined with vector database technology to query your personal document collections through an elegant Streamlit web interface. The system supports multiple AI backends including OpenAI's GPT models, Google's Gemini, and locally-hosted LLama models (GGUF format). Compatible file formats include PDF, TXT, CSV, and DOCX documents.

Built with Langchain, LlamaCPP, Streamlit, ChromaDB and Sentence Transformers. Watch the demonstration below:





<div align="center">
<video src="https://github.com/user-attachments/assets/f70695ce-17f1-4fac-93e8-8c636fd05245" controls width="600"></video>
</div>




<h1 align="center"> Getting Started 🚶 </h1>

Begin by preparing your development environment. We strongly recommend creating a virtual environment before installing the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Setting Up .env

InfoGPT uses environment variables for easy configuration. You need to set up these variables before using the application.

**For Google Gemini (Recommended):**
1. Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Add it to your `.env` file as `GOOGLE_API_KEY`

**For OpenAI GPT:**
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add it to your `.env` file as `OPENAI_API_KEY`

**For Local Models:**
- No API key required, but you need to download model files (see Local LLM section)

Copy the `example.env` template into `.env`
```bash
cp example.env .env
```
These are the environment variables you can configure:

```bash
# API Keys for AI Models
OPENAI_API_KEY=""          # For OpenAI GPT models
GOOGLE_API_KEY=""          # For Google Gemini models

# Local LLM Parameters
N_CTX=4096                 # Context window size
N_GPU_LAYERS=8            # GPU layers for acceleration
N_BATCH=512               # Batch processing size

# Vector Database Settings
RETRIEVER_K=5             # Number of documents to retrieve

# Document Processing
CHUNK_SIZE=1000           # Text chunk size for processing
CHUNK_OVERLAP=200         # Overlap between chunks

# Embedding Model
EMBEDING_MODEL="thenlper/gte-base"  # Semantic embedding model
```

### Configuring Local Language Models

To run AI models locally on your machine, download compatible GGUF format models and place them in the project's `models/` directory. Below are our top recommended models for optimal performance:

<div align="center">

| Model Name                  | Size | Performance | Hugging Face Link                                       |
|-----------------------------|------|-------------|---------------------------------------------------------|
| Llama-3.2-3B-Instruct-GGUF | 3B   | Fast        | [Link](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Llama-3.1-8B-Instruct-GGUF | 8B   | Balanced    | [Link](https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF) |
| Qwen2.5-7B-Instruct-GGUF   | 7B   | Excellent   | [Link](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF) |

Once you copy the gguf file to the models folder you are ready to use the GUI.

</div>

The beauty of the GGUF file is that it supports both CPU and GPU inference. By default, if you install dependencies it will work on CPU only but if you want it to work on GPU, you have to reinstall llama-cpp-python using the below command. Please refer to this long-chain doc for more reference 
- [Langchain Integration with LLAMAs (LLAMACPP) Documentation](https://python.langchain.com/docs/integrations/llms/llamacpp)

```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### Indexing Your Documents

InfoGPT accepts multiple document formats including PDF, TXT, CSV, DOCX, and others. To process your documents for semantic search, place them in the `document_sources` directory.

After adding your documents to the `document_sources` folder, run the indexing process to embed your content into the ChromaDB vector database:

```
python3 ingest.py
```

### Launching the Web Interface

InfoGPT features a Streamlit-powered web interface. Start the application using the launch script:

```
python3 run.py
```

This command initializes a local web server - monitor the startup process in your terminal for any messages.

The application's sidebar provides AI model selection options:
- Choose between 'Gemini', 'Local LLM', or 'OpenAI' engines
- When selecting 'Local LLM', a dropdown will appear showing all available model files in your models directory
- Select your preferred model and begin querying your documents



from util import StreamHandler, PrintRetrievalHandler, get_parent_dir
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ChatMessage
import streamlit as st
import os
import logging
import random
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logging.basicConfig(
    filename='document_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom CSS for better UI
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    
    .model-card {
        background: linear-gradient(145deg, #f0f4f8, #ffffff);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .chat-container {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #155724;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #721c24;
    }
    
    .header-container {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_app():
    # Initialize the app, including model setup, retriever, and other initializations
    global llm, qa_chain

    # Initialization code for model and retriever

    model_name = os.environ.get("EMBEDING_MODEL", "thenlper/gte-base")
    # set True to compute cosine similarity
    encode_kwargs = {'normalize_embeddings': True}

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model_norm = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    # Here is the new embeddings being used
    embedding = model_norm

    parent_directory = get_parent_dir()
    persist_directory = os.path.join(os.getcwd(), 'db')

    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    # Dynamic retrieval count - randomly choose between 3, 4, 5, and 6
    retriever_k = random.choice([3, 4, 5, 6])
    retriever = vectordb.as_retriever(search_kwargs={"k": retriever_k})
    

    # Initialization code for Streamlit sidebar and default message
    # ...

    with st.sidebar:
        st.markdown("### 🤖 Choose Your AI Model")
        st.markdown("---")
        
        option = st.selectbox(
            "Select AI Engine:",
            ("🧠 Gemini", "💻 Local LLM"),
            help="Choose between Google's Gemini or a local language model"
        )
        
        st.markdown("---")

        model_dir = os.path.join(os.getcwd(), "models")

        model_file_bin = []

        for filename in os.listdir(model_dir):
            if filename.endswith(".bin") or filename.endswith(".gguf"):
                model_file_bin.append(filename)

        if load_dotenv():
            st.success("✅ Environment configuration loaded")
            logging.info(".env has been found")

        if (option == "💻 Local LLM"):
            st.markdown("#### 🔧 Local Model Configuration")
            n_ctx = int(os.environ.get("N_CTX","2048"))
            n_gpu_layers = int(os.environ.get("N_GPU_LAYERS","8"))
            n_batch = int(os.environ.get("N_BATCH","100"))

            if not model_file_bin:
                st.error("❌ No model files found! Please add .bin or .gguf files to the models folder.")
                st.info("💡 Download models from Hugging Face and place them in the 'models' directory")
            else:
                try:
                    local_llm_file = st.selectbox(
                        "📁 Select Model File:", model_file_bin,
                        help="Choose a GGUF or BIN model file")
                    model_path = os.path.join(model_dir, local_llm_file)

                    with st.spinner("🔄 Loading local model..."):
                        llm = LlamaCpp(model_path=model_path, temperature=0, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                                       n_batch=n_batch, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
                    
                    st.markdown(f'<div class="success-box">🎉 <strong>Local LLM loaded successfully!</strong><br/>Model: {local_llm_file}</div>', unsafe_allow_html=True)
                    logging.info(f"Local LLM: {local_llm_file} has been loaded")
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <strong>Failed to load model</strong><br/>{str(e)}</div>', unsafe_allow_html=True)
                    logging.error(f"Failed to load model: {e}")
                    st.stop()

        else:  # Gemini
            st.markdown("#### 🧠 Gemini Configuration")
            try:
                with st.spinner("🔄 Connecting to Gemini..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.0,
                        streaming=True,
                        callbacks=[StreamingStdOutCallbackHandler()]
                    )
                st.markdown('<div class="success-box">🎉 <strong>Gemini 2.5 Flash connected!</strong><br/>Ready to chat with Google\'s AI</div>', unsafe_allow_html=True)
                logging.info("Gemini 2.5 Flash model has been loaded")
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ <strong>Gemini connection failed</strong><br/>Please check your GOOGLE_API_KEY in .env file</div>', unsafe_allow_html=True)
                logging.error(f"Failed to load Gemini model: {e}")
                st.stop()

    template = """
    You are a helpful, respectful, and honest assistant dedicated to providing informative and accurate response based on provided context((delimited by <ctx></ctx>)) only. You don't derive
    answer outside context, while answering your answer should be precise, accurate, clear and should not be verbose and only contain answer. In context you will have texts which is unrelated to question,
    please ignore that context only answer from the related context only.
    If the question is unclear, incoherent, or lacks factual basis, please clarify the issue rather than generating inaccurate information.

    If formatting, such as bullet points, numbered lists, tables, or code blocks, is necessary for a comprehensive response, please apply the appropriate formatting.

    <ctx>
    CONTEXT:
    {context}
    </ctx>

    QUESTION:
    {question}

    ANSWER
    """

    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Add sidebar information panel (at the end of initialize_app)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 System Info")
        try:
            doc_count = vectordb._collection.count()
            st.metric("📄 Documents Indexed", doc_count)
        except:
            st.info("📊 Database statistics unavailable")
        
        # Show dynamic range instead of fixed count
        st.metric("🔍 Retrieval Count", "3-6 chunks (dynamic)")
        
        # Add help section
        st.markdown("### 💡 Tips")
        st.info("""
• Ask specific questions about your documents
• Try: "What is the main topic?"
• Use keywords from your documents
• Be clear and concise
        """)


def process_user_message(prompt):
    # Process a user message and return the assistant's response
    global qa_chain
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.spinner("Processing..."):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty(), retrieval_handler=retrieval_handler)

        try:
            response = qa_chain.invoke(
                prompt, config={"callbacks": [retrieval_handler, stream_handler]})

            st.session_state.messages.append(
                ChatMessage(role="assistant", content=response))
            
            # Ensure retrieval status is completed after response
            retrieval_handler.finish()
            
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            retrieval_handler.finish()  # Ensure status is completed even on error
        
        # logging.info(f"User prompt: {prompt}")
        # logging.info(f"Assistant response: {response}")


def main():
    # Load custom CSS
    load_custom_css()
    
    # Page title with custom styling
    st.markdown("""
    <div class="header-container">
        <h1>🔍 VectorQuery</h1>
        <p style="font-size: 18px; margin: 0;">Intelligent Document Q&A System</p>
        <p style="font-size: 14px; margin: 5px 0 0 0; opacity: 0.9;">Ask questions about your documents using AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(
            role="assistant", content="👋 Hello! I'm VectorQuery. How can I help you explore your documents today?")]

    # Chat container with styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg.role == "assistant":
            with st.chat_message(msg.role, avatar="🤖"):
                st.write(msg.content)
        else:
            with st.chat_message(msg.role, avatar="👤"):
                st.write(msg.content)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about your documents..."):
        process_user_message(prompt)


if __name__ == "__main__":
    load_dotenv()
    initialize_app()
    main()

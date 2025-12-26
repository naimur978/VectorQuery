#!/usr/bin/env python3

import os
import sys
import random
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

def test_pdf_retrieval():
    """Test if PDFs are properly indexed and can answer questions"""
    
    # Load environment variables
    load_dotenv()
    
    print("Testing PDF retrieval and QA...")
    print("=" * 50)
    
    # Initialize embeddings (same as in ui.py)
    model_name = os.environ.get("EMBEDING_MODEL", "thenlper/gte-base")
    encode_kwargs = {'normalize_embeddings': True}
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Embedding model: {model_name}")
    
    try:
        model_norm = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs=encode_kwargs
        )
        print("✅ Embeddings loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return
    
    # Load vector database
    persist_directory = os.path.join(os.getcwd(), 'db')
    
    if not os.path.exists(persist_directory):
        print(f"❌ Database directory not found: {persist_directory}")
        print("Run 'python3 ingest.py' first to create the database")
        return
    
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=model_norm)
        print("✅ Vector database loaded successfully")
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return
    
    # Test retrieval
    test_question = "Where do most heart disease deaths happen globally?"
    print(f"\n🔍 Testing question: '{test_question}'")
    print("-" * 50)
    
    try:
        # Dynamic retrieval count - randomly choose between 3, 4, 5, and 6
        retriever_k = random.choice([3, 4, 5, 6])
        retriever = vectordb.as_retriever(search_kwargs={"k": retriever_k})
        
        # Retrieve relevant documents
        docs = retriever.invoke(test_question)
        
        if not docs:
            print("❌ No relevant documents found")
            print("This might mean:")
            print("  1. PDFs weren't processed properly")
            print("  2. PDFs don't contain information about heart disease")
            print("  3. The question doesn't match the document content")
            return
        
        print(f"✅ Found {len(docs)} relevant document chunks")
        print("\n📄 Retrieved content:")
        print("=" * 50)
        
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document Chunk {i} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:300]}...")
            if len(doc.page_content) > 300:
                print("(truncated)")
            print()
        
        # Test with a simple Q&A chain
        print("\n🤖 Testing with OpenAI (if available)...")
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough
            
            # Use same template as ui.py
            template = """
            You are a helpful, respectful, and honest assistant dedicated to providing informative and accurate response based on provided context((delimited by <ctx></ctx>)) only. You don't derive
            answer outside context, while answering your answer should be precise, accurate, clear and should not be verbose and only contain answer.

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
            
            llm = ChatOpenAI(temperature=0.0)
            
            qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = qa_chain.invoke(test_question)
            print("✅ AI Response:")
            print("-" * 30)
            print(response)
            
        except Exception as e:
            print(f"⚠️  Could not test with OpenAI: {e}")
            print("But document retrieval is working!")
            
    except Exception as e:
        print(f"❌ Error during retrieval test: {e}")
        return
    
    print(f"\n✅ PDF testing completed successfully!")
    print("Your VectorQuery should be able to answer questions from your PDFs.")

if __name__ == "__main__":
    test_pdf_retrieval()
#!/usr/bin/env python3

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from dotenv import load_dotenv

def check_database_content():
    """Check what documents are actually stored in the vector database"""
    
    load_dotenv()
    
    print("Checking vector database content...")
    print("=" * 50)
    
    # Initialize embeddings
    model_name = os.environ.get("EMBEDING_MODEL", "thenlper/gte-base")
    encode_kwargs = {'normalize_embeddings': True}
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    model_norm = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    
    # Load vector database
    persist_directory = os.path.join(os.getcwd(), 'db')
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=model_norm)
    
    # Get all documents
    try:
        # Get collection info
        collection = vectordb._collection
        count = collection.count()
        print(f"Total document chunks in database: {count}")
        
        # Get a sample of documents to see what sources we have
        results = collection.get(include=['documents', 'metadatas'], limit=100)
        
        # Collect unique sources
        sources = set()
        for metadata in results['metadatas']:
            if metadata and 'source' in metadata:
                sources.add(metadata['source'])
        
        print(f"\nUnique document sources ({len(sources)}):")
        print("-" * 40)
        for source in sorted(sources):
            source_name = os.path.basename(source)
            print(f"  📄 {source_name}")
            
        # Check for PDF files specifically
        pdf_sources = [s for s in sources if s.lower().endswith('.pdf')]
        print(f"\n📕 PDF files found: {len(pdf_sources)}")
        for pdf in pdf_sources:
            print(f"  • {os.path.basename(pdf)}")
            
        if not pdf_sources:
            print("\n❌ No PDF files found in the database!")
            print("This might mean:")
            print("  1. PDFs weren't processed during ingestion")
            print("  2. PDFs are empty or couldn't be read")
            print("  3. There's an issue with the PDF loader")
            
            # Check what's in document_sources
            doc_dir = "document_sources"
            if os.path.exists(doc_dir):
                print(f"\n📁 Files in {doc_dir}:")
                for file in os.listdir(doc_dir):
                    print(f"  • {file}")
            
        print(f"\n✅ Database check completed!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database_content()
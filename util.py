from langchain_core.callbacks import BaseCallbackHandler
import os
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", retrieval_handler=None):
        self.container = container
        self.text = initial_text
        self.retrieval_handler = retrieval_handler

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM generation ends - ensure retrieval status is complete"""
        if self.retrieval_handler:
            self.retrieval_handler.finish()

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")
        self.documents_found = False

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** Searching...")

    def on_retriever_end(self, documents, **kwargs):
        self.documents_found = True
        self.status.update(label=f"**Context Retrieval:** Found {len(documents)} relevant documents")
        
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx + 1} from {source}**")
            self.status.markdown(doc.page_content)
            self.status.write(f"-------------------------------------")
        
        # Mark as complete after displaying documents
        self.status.update(state="complete")
    
    def finish(self):
        """Method to manually finish the retrieval process"""
        if hasattr(self.status, 'update'):
            self.status.update(state="complete")


def get_parent_dir():
    script_directory = os.path.dirname(__file__)
    # Get the parent directory
    parent_directory = os.path.dirname(script_directory)
    return parent_directory



from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language

from langchain_community.document_loaders import PyPDFLoader

text= '''
from typing import List, Dict, Any

class RAGPipeline:
    """
    A sample class representing a basic Retrieval-Augmented Generation pipeline.
    Demonstrates standard Python object-oriented programming (OOP) practices.
    """
    
    # Class attribute (shared across all instances)
    PIPELINE_VERSION: str = "1.0.0"

    def __init__(self, vector_db_name: str, embedding_model: str):
        """
        Constructor method to initialize instance attributes.
        """
        # Instance attributes (unique to each object)
        self.vector_db_name: str = vector_db_name
        self.embedding_model: str = embedding_model
        self.is_connected: bool = False
        
        # Internal state variable (protected/private by convention)
        self._document_count: int = 0

    def connect(self) -> bool:
        """Connects to the specified vector database."""
        print(f"Connecting to {self.vector_db_name} using {self.embedding_model}...")
        # Simulate connection logic
        self.is_connected = True
        return self.is_connected

    def ingest_documents(self, documents: List[str]) -> int:
        """Processes and increments the internal document count."""
        if not self.is_connected:
            raise ConnectionError("Must connect to the database before ingesting data.")
        
        for doc in documents:
            self._document_count += 1
            self._log_ingestion(doc)
            
        return self._document_count

    def _log_ingestion(self, document_preview: str) -> None:
        """
        A private helper method (indicated by a single leading underscore).
        Used internally and not meant to be called directly from outside the class.
        """
        # Truncate string for logging purposes
        preview = document_preview[:20] + "..." if len(document_preview) > 20 else document_preview
        print(f"[LOG] successfully processed: '{preview}'")

    @property
    def total_documents(self) -> int:
        """A property decorator that allows reading a value like an attribute."""
        return self._document_count

    @classmethod
    def create_default_pipeline(cls) -> "RAGPipeline":
        """
        A class method acting as an alternative constructor.
        It instantiates the class using default production settings.
        """
        return cls(vector_db_name="ChromaDB", embedding_model="text-embedding-3-small")


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate the class using the default factory method
    pipeline = RAGPipeline.create_default_pipeline()
    
    # 2. Access class and instance attributes
    print(f"Pipeline Version: {pipeline.PIPELINE_VERSION}")
    print(f"Target DB: {pipeline.vector_db_name}")

    # 3. Call instance methods
    pipeline.connect()
    
    sample_docs = [
        "LangGraph is excellent for building cyclical agent workflows.",
        "ColPali uses vision models to index entire document pages directly."
    ]
    
    pipeline.ingest_documents(sample_docs)
    
    # 4. Access data cleanly via the @property getter
    print(f"Total documents managed by this instance: {pipeline.total_documents}")

'''



splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

result= splitter.split_text(text)


print(len(result))
print(result)

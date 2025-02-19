"""
Base classes and utilities for LangChain pipelines.
"""

import os
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

class BaseLLMPipeline:
    """Base class for LangChain pipelines with common LLM initialization."""
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """Initialize the base pipeline with LLM configuration.
        
        Args:
            model_name: The OpenAI model to use
            temperature: The temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.llm = ChatOpenAI(
            model_name=model_name or os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
            temperature=temperature or float(os.getenv('TEMPERATURE', 0.7)),
            max_tokens=max_tokens or int(os.getenv('MAX_TOKENS', 2000))
        )

class VectorStoreManager:
    """Manages vector store operations for document similarity search."""
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the vector store manager.
        
        Args:
            embeddings: The embeddings model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        
    def create_vector_store(
        self,
        texts: List[str],
        collection_name: Optional[str] = None
    ) -> FAISS:
        """Create a vector store from a list of texts.
        
        Args:
            texts: List of text strings to embed
            collection_name: Name for the collection
            
        Returns:
            The created FAISS vector store
        """
        collection_name = collection_name or os.getenv('COLLECTION_NAME', 'default_collection')
        self.vector_store = FAISS.from_texts(
            texts,
            self.embeddings,
            collection_name
        )
        return self.vector_store
        
    def load_and_split_document(self, file_path: str) -> List[Any]:
        """Load and split a document into chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
        
    def query_vector_store(self, query: str, k: int = 4) -> List[Any]:
        """Query the vector store for similar documents.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        return self.vector_store.similarity_search(query, k=k)
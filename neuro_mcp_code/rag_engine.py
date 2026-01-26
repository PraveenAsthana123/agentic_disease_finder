"""
RAG Engine for arXiv Paper Downloader
Integrates Ollama, ChromaDB, and advanced RAG features
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib

import chromadb
from chromadb.config import Settings
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    """Advanced RAG engine with Ollama and ChromaDB"""

    def __init__(self,
                 collection_name: str = "arxiv_papers",
                 embed_model: str = "nomic-embed-text",
                 chat_model: str = "llama3.2:3b"):
        """
        Initialize RAG engine

        Args:
            collection_name: Name of ChromaDB collection
            embed_model: Ollama embedding model
            chat_model: Ollama chat model
        """
        self.embed_model = embed_model
        self.chat_model = chat_model

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "arXiv research papers"}
            )
            print(f"✓ Created new collection: {collection_name}")

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Cache for embeddings and responses
        self.embedding_cache = {}
        self.response_cache = {}

        # Conversation history
        self.conversation_history = []

        print(f"✓ RAG Engine initialized")
        print(f"  - Embedding model: {self.embed_model}")
        print(f"  - Chat model: {self.chat_model}")

    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for text using Ollama

        Args:
            text: Input text
            use_cache: Whether to use cached embeddings

        Returns:
            List of floats representing the embedding
        """
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            response = ollama.embeddings(
                model=self.embed_model,
                prompt=text
            )
            embedding = response['embedding']

            # Cache the embedding
            if use_cache:
                self.embedding_cache[cache_key] = embedding

            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split document into chunks

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.split_text(text)

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'text': chunk,
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })

        return chunked_docs

    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text = "\n\n".join([page.page_content for page in pages])
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    def ingest_pdf(self, pdf_path: str, metadata: Dict) -> Dict:
        """
        Ingest a PDF into the vector database

        Args:
            pdf_path: Path to PDF file
            metadata: Paper metadata

        Returns:
            Ingestion statistics
        """
        start_time = time.time()

        # Extract text
        text = self.extract_pdf_text(pdf_path)

        if not text:
            return {
                'success': False,
                'error': 'Failed to extract text from PDF'
            }

        # Chunk document
        chunks = self.chunk_document(text, metadata)

        # Generate embeddings and add to database
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{metadata.get('arxiv_id', 'unknown')}_{chunk['metadata']['chunk_id']}"
            embedding = self.get_embedding(chunk['text'])

            if embedding:
                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(chunk['text'])
                metadatas.append(chunk['metadata'])

        # Add to ChromaDB
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

        elapsed_time = time.time() - start_time

        return {
            'success': True,
            'pdf_path': pdf_path,
            'chunks': len(chunks),
            'elapsed_time': elapsed_time,
            'arxiv_id': metadata.get('arxiv_id', 'unknown')
        }

    def pre_retrieval_filter(self,
                            query: str,
                            filters: Optional[Dict] = None) -> Dict:
        """
        Pre-retrieval filtering based on metadata

        Args:
            query: Search query
            filters: Metadata filters (e.g., date range, authors)

        Returns:
            Processed filters
        """
        where_filter = {}

        if filters:
            # Date range filtering
            if 'date_from' in filters:
                where_filter['published'] = {'$gte': filters['date_from']}

            if 'date_to' in filters:
                if 'published' in where_filter:
                    where_filter['published']['$lte'] = filters['date_to']
                else:
                    where_filter['published'] = {'$lte': filters['date_to']}

            # Author filtering
            if 'authors' in filters:
                where_filter['authors'] = {'$contains': filters['authors']}

            # Topic/Folder filtering
            if 'folder' in filters:
                where_filter['folder'] = filters['folder']

        return where_filter

    def post_retrieval_rerank(self,
                              query: str,
                              results: List[Dict],
                              top_k: int = 5) -> List[Dict]:
        """
        Post-retrieval reranking using cosine similarity

        Args:
            query: Search query
            results: Retrieved results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate cosine similarity scores
        for result in results:
            result_embedding = result.get('embedding', [])
            if result_embedding and query_embedding:
                similarity = cosine_similarity(
                    [query_embedding],
                    [result_embedding]
                )[0][0]
                result['rerank_score'] = float(similarity)
            else:
                result['rerank_score'] = 0.0

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]

    def retrieve(self,
                query: str,
                n_results: int = 5,
                filters: Optional[Dict] = None,
                use_reranking: bool = True) -> List[Dict]:
        """
        Retrieve relevant documents from vector database

        Args:
            query: Search query
            n_results: Number of results to retrieve
            filters: Pre-retrieval filters
            use_reranking: Whether to apply post-retrieval reranking

        Returns:
            List of relevant documents
        """
        # Pre-retrieval filtering
        where_filter = self.pre_retrieval_filter(query, filters) if filters else None

        # Get query embedding
        query_embedding = self.get_embedding(query)

        if not query_embedding:
            return []

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2 if use_reranking else n_results,
                where=where_filter if where_filter else None
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'embedding': results['embeddings'][0][i] if results.get('embeddings') and results['embeddings'][0] else []
                })

            # Post-retrieval reranking
            if use_reranking:
                formatted_results = self.post_retrieval_rerank(query, formatted_results, n_results)

            return formatted_results

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate_response(self,
                         query: str,
                         context: List[Dict],
                         model: Optional[str] = None,
                         use_cache: bool = True) -> Dict:
        """
        Generate response using Ollama with retrieved context

        Args:
            query: User query
            context: Retrieved context documents
            model: Ollama model to use (defaults to self.chat_model)
            use_cache: Whether to use cached responses

        Returns:
            Response dictionary with answer and metadata
        """
        # Check cache
        cache_key = hashlib.md5(f"{query}_{model}".encode()).hexdigest()
        if use_cache and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            cached['cached'] = True
            return cached

        # Build context from retrieved documents
        context_text = "\n\n".join([
            f"[Source: {doc['metadata'].get('title', 'Unknown')}]\n{doc['document']}"
            for doc in context
        ])

        # Build conversation history
        history_text = "\n".join([
            f"User: {msg['query']}\nAssistant: {msg['response']}"
            for msg in self.conversation_history[-3:]  # Last 3 exchanges
        ])

        # Create prompt
        prompt = f"""You are a helpful research assistant with access to arXiv papers.
Answer the user's question based on the provided context.

Conversation History:
{history_text if history_text else "No previous conversation"}

Context from research papers:
{context_text}

User Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain relevant information, say so."""

        # Generate response
        try:
            start_time = time.time()

            response = ollama.generate(
                model=model or self.chat_model,
                prompt=prompt
            )

            elapsed_time = time.time() - start_time

            result = {
                'answer': response['response'],
                'model': model or self.chat_model,
                'context_used': len(context),
                'elapsed_time': elapsed_time,
                'cached': False,
                'sources': [doc['metadata'].get('title', 'Unknown') for doc in context]
            }

            # Cache the response
            if use_cache:
                self.response_cache[cache_key] = result

            return result

        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'error': True
            }

    def chat(self,
            query: str,
            model: Optional[str] = None,
            n_context: int = 5,
            filters: Optional[Dict] = None,
            use_cache: bool = True) -> Dict:
        """
        Main chat interface combining retrieval and generation

        Args:
            query: User query
            model: Ollama model to use
            n_context: Number of context documents to retrieve
            filters: Pre-retrieval filters
            use_cache: Whether to use caching

        Returns:
            Response with answer and metadata
        """
        # Retrieve relevant context
        context = self.retrieve(query, n_results=n_context, filters=filters)

        # Generate response
        response = self.generate_response(query, context, model=model, use_cache=use_cache)

        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'response': response['answer'],
            'timestamp': datetime.now().isoformat(),
            'model': model or self.chat_model
        })

        # Add retrieval info to response
        response['retrieval_count'] = len(context)
        response['context'] = context

        return response

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name,
                'embedding_cache_size': len(self.embedding_cache),
                'response_cache_size': len(self.response_cache),
                'conversation_length': len(self.conversation_history)
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_caches(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.conversation_history.clear()
        print("✓ Caches cleared")

    def list_available_models(self) -> Dict:
        """List available Ollama models"""
        try:
            models_response = ollama.list()
            return {
                'success': True,
                'models': [model.model for model in models_response.models]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

"""
Integration with LangChain and LlamaIndex for AI orchestration.
"""

from typing import List, Dict, Any, Optional, Callable
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools import BaseTool
from langchain.agents import Tool
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import NodeWithScore, TextNode
import numpy as np


class SIEXEmbeddings(Embeddings):
    """LangChain-compatible embeddings using SIE-X engine."""

    def __init__(self, engine: 'SemanticIntelligenceEngine'):
        self.engine = engine

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using SIE-X."""
        embeddings = self.engine._generate_embeddings_batch(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query using SIE-X."""
        embedding = self.engine._generate_embeddings_batch([text])[0]
        return embedding.tolist()


class SIEXTextSplitter(TextSplitter):
    """LangChain-compatible text splitter using SIE-X chunking."""

    def __init__(self, engine: 'SemanticIntelligenceEngine'):
        self.engine = engine
        super().__init__()

    def split_text(self, text: str) -> List[str]:
        """Split text using SIE-X chunking strategy."""
        chunks = self.engine.chunker.chunk(text)
        return chunks


class SIEXVectorStore(VectorStore):
    """LangChain-compatible vector store using SIE-X."""

    def __init__(self, engine: 'SemanticIntelligenceEngine'):
        self.engine = engine
        self.documents: List[Document] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict]] = None,
            **kwargs
    ) -> List[str]:
        """Add texts to vector store."""
        ids = []
        for i, text in enumerate(texts):
            doc_id = f"doc_{len(self.documents)}"
            metadata = metadatas[i] if metadatas else {}

            # Extract keywords and embeddings
            keywords = self.engine.extract(text, top_k=20)

            # Store document
            doc = Document(
                page_content=text,
                metadata={
                    **metadata,
                    "keywords": [kw.text for kw in keywords],
                    "keyword_scores": {kw.text: kw.score for kw in keywords}
                }
            )
            self.documents.append(doc)

            # Cache embeddings
            for kw in keywords:
                if kw.embeddings is not None:
                    self.embeddings_cache[kw.text] = kw.embeddings

            ids.append(doc_id)

        return ids

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs
    ) -> List[Document]:
        """Search similar documents using SIE-X semantic search."""
        # Extract query keywords
        query_keywords = self.engine.extract(query, top_k=10)
        query_embedding = self.engine._generate_embeddings_batch([query])[0]

        # Score documents
        doc_scores = []
        for doc in self.documents:
            # Keyword overlap score
            doc_keywords = set(doc.metadata.get("keywords", []))
            query_kw_texts = set([kw.text for kw in query_keywords])
            overlap_score = len(doc_keywords & query_kw_texts) / max(len(doc_keywords), len(query_kw_texts))

            # Semantic similarity score
            doc_embedding = self.engine._generate_embeddings_batch([doc.page_content])[0]
            semantic_score = float(np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            ))

            # Combined score
            total_score = 0.3 * overlap_score + 0.7 * semantic_score
            doc_scores.append((doc, total_score))

        # Sort and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:k]]

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs
    ) -> "SIEXVectorStore":
        """Create vector store from texts."""
        # Extract engine from embeddings
        if isinstance(embedding, SIEXEmbeddings):
            store = cls(embedding.engine)
            store.add_texts(texts, metadatas)
            return store
        else:
            raise ValueError("SIEXVectorStore requires SIEXEmbeddings")


class SIEXRetriever(BaseRetriever):
    """LangChain retriever using SIE-X for intelligent retrieval."""

    def __init__(
            self,
            engine: 'SemanticIntelligenceEngine',
            vector_store: SIEXVectorStore,
            use_keywords: bool = True,
            use_clustering: bool = True,
            rerank: bool = True
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.use_keywords = use_keywords
        self.use_clustering = use_clustering
        self.rerank = rerank

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents using SIE-X enhanced retrieval."""
        # Initial retrieval
        initial_docs = self.vector_store.similarity_search(query, k=20)

        if not self.rerank:
            return initial_docs[:4]

        # Extract query keywords for enhanced ranking
        query_keywords = self.engine.extract(query, top_k=15)
        query_concepts = {kw.text: kw for kw in query_keywords}

        # Re-rank documents using advanced features
        reranked_docs = []
        for doc in initial_docs:
            score = 0.0

            # Keyword relevance
            if self.use_keywords:
                doc_keywords = doc.metadata.get("keywords", [])
                keyword_scores = doc.metadata.get("keyword_scores", {})

                for kw in doc_keywords:
                    if kw in query_concepts:
                        # Boost by both query and document keyword importance
                        score += query_concepts[kw].score * keyword_scores.get(kw, 0)

            # Semantic clustering bonus
            if self.use_clustering:
                # Check if document keywords belong to same semantic clusters
                doc_text = doc.page_content
                doc_keywords_full = self.engine.extract(doc_text, top_k=20)

                cluster_overlap = 0
                for doc_kw in doc_keywords_full:
                    for query_kw in query_keywords:
                        if (doc_kw.semantic_cluster is not None and
                                query_kw.semantic_cluster is not None and
                                doc_kw.semantic_cluster == query_kw.semantic_cluster):
                            cluster_overlap += 1

                score += cluster_overlap * 0.1

            reranked_docs.append((doc, score))

        # Sort by score and return top results
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs[:4]]

    class SIEXKeywordTool(BaseTool):
        """LangChain tool for keyword extraction using SIE-X."""

        name = "sie_x_keyword_extractor"
        description = (
            "Extract semantic keywords from text. "
            "Useful for understanding key concepts, entities, and topics. "
            "Input should be the text to analyze."
        )

        def __init__(self, engine: 'SemanticIntelligenceEngine'):
            super().__init__()
            self.engine = engine

        def _run(self, text: str, run_manager: Optional = None) -> str:
            """Run keyword extraction."""
            keywords = self.engine.extract(text, top_k=15, output_format='json')
            return json.dumps(keywords, indent=2)

        async def _arun(self, text: str, run_manager: Optional = None) -> str:
            """Run keyword extraction asynchronously."""
            keywords = await self.engine.extract_async(text, top_k=15, output_format='json')
            return json.dumps(keywords, indent=2)

    class SIEXMultiDocTool(BaseTool):
        """LangChain tool for multi-document analysis using SIE-X."""

        name = "sie_x_multi_doc_analyzer"
        description = (
            "Analyze multiple documents to find common themes and distinctive elements. "
            "Input should be a JSON array of document texts."
        )

        def __init__(self, engine: 'SemanticIntelligenceEngine'):
            super().__init__()
            self.engine = engine

        def _run(self, documents_json: str, run_manager: Optional = None) -> str:
            """Run multi-document analysis."""
            documents = json.loads(documents_json)
            if not isinstance(documents, list):
                return "Error: Input must be a JSON array of document texts"

            results = asyncio.run(self.engine.extract_multiple_advanced(
                texts=documents,
                top_k_common=10,
                top_k_distinctive=5
            ))

            return json.dumps(results, indent=2)

    # LlamaIndex Integration
    class SIEXNodeParser(NodeParser):
        """LlamaIndex node parser using SIE-X chunking."""

        def __init__(self, engine: 'SemanticIntelligenceEngine'):
            self.engine = engine
            super().__init__()

        def get_nodes_from_documents(
                self,
                documents: List[Document],
                show_progress: bool = False,
                **kwargs
        ) -> List[TextNode]:
            """Parse documents into nodes using SIE-X."""
            nodes = []

            for doc_idx, doc in enumerate(documents):
                # Chunk document
                chunks = self.engine.chunker.chunk(doc.text)

                for chunk_idx, chunk in enumerate(chunks):
                    # Extract keywords for chunk
                    keywords = self.engine.extract(chunk, top_k=10)

                    # Create node
                    node = TextNode(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_idx,
                            "keywords": [kw.text for kw in keywords],
                            "keyword_scores": {kw.text: kw.score for kw in keywords},
                            "doc_id": doc.doc_id or f"doc_{doc_idx}"
                        },
                        relationships={
                            "SOURCE": doc.doc_id or f"doc_{doc_idx}"
                        }
                    )
                    nodes.append(node)

            return nodes

    class SIEXQueryEngine:
        """Advanced query engine combining SIE-X with LLMs."""

        def __init__(
                self,
                engine: 'SemanticIntelligenceEngine',
                llm: Any,  # LangChain LLM
                vector_store: Optional[SIEXVectorStore] = None
        ):
            self.engine = engine
            self.llm = llm
            self.vector_store = vector_store or SIEXVectorStore(engine)
            self.retriever = SIEXRetriever(engine, self.vector_store)

        async def query(
                self,
                question: str,
                use_keywords: bool = True,
                use_context: bool = True
        ) -> Dict[str, Any]:
            """Answer question using SIE-X enhanced retrieval."""
            # Extract question keywords for better understanding
            question_keywords = await self.engine.extract_async(question, top_k=10)

            # Build enhanced query
            enhanced_query = question
            if use_keywords:
                key_concepts = [kw.text for kw in question_keywords[:5]]
                enhanced_query = f"{question}\nKey concepts: {', '.join(key_concepts)}"

            # Retrieve relevant documents
            if use_context:
                relevant_docs = self.retriever.get_relevant_documents(enhanced_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Build prompt with context
                prompt = f"""Answer the following question based on the provided context.

        Context:
        {context}

        Question: {question}

        Answer:"""
            else:
                prompt = question

            # Get LLM response
            response = await self.llm.agenerate([prompt])
            answer = response.generations[0][0].text

            # Extract answer keywords for summary
            answer_keywords = await self.engine.extract_async(answer, top_k=10)

            return {
                "question": question,
                "answer": answer,
                "question_keywords": [kw.to_dict() for kw in question_keywords],
                "answer_keywords": [kw.to_dict() for kw in answer_keywords],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ] if use_context else []
            }

"""
Retrieval Models Module
Implements various retrieval models: TF-IDF, Cosine Similarity, Boolean AND
"""

import math
from collections import Counter


class RetrievalModels:
    """Container class for different retrieval models"""
    
    def __init__(self, inverted_index):
        """
        Initialize retrieval models.
        
        Args:
            inverted_index: InvertedIndex instance
        """
        self.index = inverted_index
    
    def tf_idf_score(self, query_terms, doc_id):
        """
        Calculate TF-IDF score for a document given a query.
        
        TF-IDF Formula:
        score = Σ (1 + log(tf)) * log(N / df)
        
        Args:
            query_terms: List of query tokens
            doc_id: Document ID
            
        Returns:
            float: TF-IDF score
        """
        score = 0.0
        N = self.index.num_docs
        
        for term in query_terms:
            # Get term frequency in document
            tf = self.index.get_term_frequency(term, doc_id)
            
            if tf > 0:
                # Get document frequency
                df = self.index.get_document_frequency(term)
                
                if df > 0:
                    # Calculate TF component (logarithmic)
                    tf_component = 1 + math.log(tf)
                    
                    # Calculate IDF component
                    idf = math.log(N / df)
                    
                    score += tf_component * idf
        
        return score
    
    def cosine_similarity(self, query_terms, doc_id):
        """
        Calculate Cosine Similarity between query and document.
        
        Cosine Similarity:
        similarity = (Q · D) / (||Q|| * ||D||)
        
        Args:
            query_terms: List of query tokens
            doc_id: Document ID
            
        Returns:
            float: Cosine similarity score
        """
        # Create query vector
        query_vec = Counter(query_terms)
        
        # Get document vector
        doc_vec = self.index.get_document_vector(doc_id)
        
        # Calculate dot product
        dot_product = sum(query_vec[term] * doc_vec.get(term, 0) 
                         for term in query_vec)
        
        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(count ** 2 for count in query_vec.values()))
        doc_magnitude = math.sqrt(sum(count ** 2 for count in doc_vec.values()))
        
        # Avoid division by zero
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def boolean_and(self, query_terms, doc_id):
        """
        Boolean AND retrieval - document must contain ALL query terms.
        
        Args:
            query_terms: List of query tokens
            doc_id: Document ID
            
        Returns:
            float: 1.0 if document contains all terms, 0.0 otherwise
        """
        for term in query_terms:
            if self.index.get_term_frequency(term, doc_id) == 0:
                return 0.0
        return 1.0


def rank_documents(query_terms, inverted_index, model='tfidf', top_k=10):
    """
    Rank documents for a query using specified model.
    
    Args:
        query_terms: List of preprocessed query tokens
        inverted_index: InvertedIndex instance
        model: Retrieval model ('tfidf', 'cosine', 'boolean')
        top_k: Number of top documents to return
        
    Returns:
        list: List of (doc_id, score) tuples, sorted by score
    """
    if not query_terms:
        return []
    
    retrieval_models = RetrievalModels(inverted_index)
    results = []
    
    # Get all document IDs
    for doc_id in inverted_index.doc_lengths.keys():
        # Calculate score based on model
        if model == 'tfidf':
            score = retrieval_models.tf_idf_score(query_terms, doc_id)
        elif model == 'cosine':
            score = retrieval_models.cosine_similarity(query_terms, doc_id)
        elif model == 'boolean':
            score = retrieval_models.boolean_and(query_terms, doc_id)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Add to results if score > 0
        if score > 0:
            results.append((doc_id, score))
    
    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results
    return results[:top_k]


def search_all_queries(processed_queries, inverted_index, model='tfidf', top_k=100):
    """
    Search all queries and return ranked results.
    
    Args:
        processed_queries: dict of {query_id: list of tokens}
        inverted_index: InvertedIndex instance
        model: Retrieval model to use
        top_k: Number of top documents per query
        
    Returns:
        dict: {query_id: list of doc_ids in ranked order}
    """
    all_results = {}
    
    for query_id, query_terms in processed_queries.items():
        # Rank documents
        ranked_docs = rank_documents(query_terms, inverted_index, model, top_k)
        
        # Extract just doc_ids
        all_results[query_id] = [doc_id for doc_id, score in ranked_docs]
    
    return all_results
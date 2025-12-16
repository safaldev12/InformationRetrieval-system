"""
Main Execution Script for IR System Evaluation
Complete pipeline: Data loading -> Model execution -> Comprehensive evaluation -> Visualization
Author: Safal Subedi
Course: TECH 400 - Introduction to Information Retrieval
Date: December 2025
"""

import re
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from enhanced_evaluation import IREvaluator
from visualization import IRVisualizer
import json
import os


class CISIDataLoader:
    """Handles loading and parsing of CISI dataset files."""
    
    @staticmethod
    def parse_cisi_documents(filepath: str) -> Dict[int, str]:
        """
        Parse CISI.ALL file to extract documents.
        
        Args:
            filepath: Path to CISI.ALL file
            
        Returns:
            Dictionary mapping doc_id to document text
        """
        documents = {}
        current_doc_id = None
        current_text = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    # New document starts with .I
                    if line.startswith('.I'):
                        # Save previous document
                        if current_doc_id is not None and current_text:
                            documents[current_doc_id] = ' '.join(current_text)
                        
                        # Start new document
                        current_doc_id = int(line.split()[1])
                        current_text = []
                    
                    # Skip field markers
                    elif line.startswith('.'):
                        continue
                    
                    # Collect document text
                    elif current_doc_id is not None and line:
                        current_text.append(line)
                
                # Save last document
                if current_doc_id is not None and current_text:
                    documents[current_doc_id] = ' '.join(current_text)
        
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath}")
            return {}
        
        return documents
    
    @staticmethod
    def parse_cisi_queries(filepath: str) -> Dict[int, str]:
        """
        Parse CISI.QRY file to extract queries.
        
        Args:
            filepath: Path to CISI.QRY file
            
        Returns:
            Dictionary mapping query_id to query text
        """
        queries = {}
        current_query_id = None
        current_text = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('.I'):
                        if current_query_id is not None and current_text:
                            queries[current_query_id] = ' '.join(current_text)
                        
                        current_query_id = int(line.split()[1])
                        current_text = []
                    
                    elif line.startswith('.W'):
                        continue
                    
                    elif current_query_id is not None and line:
                        current_text.append(line)
                
                if current_query_id is not None and current_text:
                    queries[current_query_id] = ' '.join(current_text)
        
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath}")
            return {}
        
        return queries
    
    @staticmethod
    def parse_relevance_judgments(filepath: str) -> Dict[int, Set[int]]:
        """
        Parse CISI.REL file to extract relevance judgments.
        
        Args:
            filepath: Path to CISI.REL file
            
        Returns:
            Dictionary mapping query_id to set of relevant doc_ids
        """
        relevance = defaultdict(set)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        query_id = int(parts[0])
                        doc_id = int(parts[1])
                        relevance[query_id].add(doc_id)
        
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath}")
            return {}
        
        return dict(relevance)


class TextPreprocessor:
    """Handles text preprocessing: tokenization, stopword removal, stemming."""
    
    # Common English stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Convert text to lowercase and split into tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """
        Remove common stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        return [token for token in tokens if token not in TextPreprocessor.STOPWORDS]
    
    @staticmethod
    def simple_stem(word: str) -> str:
        """
        Apply simple suffix-stripping stemming.
        
        Args:
            word: Input word
            
        Returns:
            Stemmed word
        """
        # Common suffix removal patterns
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'tion', 'ness', 'ment']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    @classmethod
    def preprocess(cls, text: str) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        tokens = cls.tokenize(text)
        tokens = cls.remove_stopwords(tokens)
        tokens = [cls.simple_stem(token) for token in tokens]
        return tokens


class InvertedIndex:
    """Builds and maintains inverted index structure."""
    
    def __init__(self):
        """Initialize inverted index."""
        self.index = defaultdict(lambda: defaultdict(int))  # {term: {doc_id: frequency}}
        self.doc_lengths = {}  # {doc_id: number of terms}
        self.doc_vectors = {}  # {doc_id: {term: count}}
    
    def build(self, documents: Dict[int, str]):
        """
        Build inverted index from documents.
        
        Args:
            documents: Dictionary mapping doc_id to document text
        """
        print("Building inverted index...")
        
        for doc_id, text in documents.items():
            # Preprocess document
            tokens = TextPreprocessor.preprocess(text)
            
            # Store document length
            self.doc_lengths[doc_id] = len(tokens)
            
            # Build term frequency map
            term_freq = defaultdict(int)
            for term in tokens:
                term_freq[term] += 1
                self.index[term][doc_id] += 1
            
            self.doc_vectors[doc_id] = dict(term_freq)
        
        print(f"Index built: {len(self.index)} unique terms, {len(documents)} documents")


class RetrievalModels:
    """Implements TF-IDF, Cosine Similarity, and Boolean AND retrieval models."""
    
    def __init__(self, index: InvertedIndex, num_documents: int):
        """
        Initialize retrieval models.
        
        Args:
            index: Inverted index structure
            num_documents: Total number of documents in collection
        """
        self.index = index
        self.N = num_documents
    
    def tfidf_retrieval(self, query: str, top_k: int = 100) -> List[int]:
        """
        TF-IDF ranking model.
        
        Formula: score = (1 + log(tf)) × log(N / df)
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of doc_ids ranked by TF-IDF score
        """
        query_terms = TextPreprocessor.preprocess(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            if term in self.index.index:
                # Document frequency for this term
                df = len(self.index.index[term])
                idf = math.log(self.N / df)
                
                # Score each document containing this term
                for doc_id, tf in self.index.index[term].items():
                    tfidf = (1 + math.log(tf)) * idf
                    scores[doc_id] += tfidf
        
        # Sort by score descending
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in ranked_docs[:top_k]]
    
    def cosine_similarity_retrieval(self, query: str, top_k: int = 100) -> List[int]:
        """
        Cosine similarity ranking model.
        
        Formula: similarity = (Q · D) / (||Q|| × ||D||)
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of doc_ids ranked by cosine similarity
        """
        query_terms = TextPreprocessor.preprocess(query)
        query_vector = defaultdict(int)
        
        # Build query vector
        for term in query_terms:
            query_vector[term] += 1
        
        # Calculate query magnitude
        query_magnitude = math.sqrt(sum(count ** 2 for count in query_vector.values()))
        
        if query_magnitude == 0:
            return []
        
        scores = {}
        
        # Calculate cosine similarity for each document
        for doc_id, doc_vector in self.index.doc_vectors.items():
            # Dot product
            dot_product = sum(query_vector[term] * doc_vector.get(term, 0) 
                            for term in query_vector)
            
            if dot_product > 0:
                # Document magnitude
                doc_magnitude = math.sqrt(sum(count ** 2 for count in doc_vector.values()))
                
                # Cosine similarity
                similarity = dot_product / (query_magnitude * doc_magnitude)
                scores[doc_id] = similarity
        
        # Sort by similarity descending
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in ranked_docs[:top_k]]
    
    def boolean_and_retrieval(self, query: str, top_k: int = 100) -> List[int]:
        """
        Boolean AND retrieval - documents must contain ALL query terms.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of doc_ids that contain all query terms
        """
        query_terms = TextPreprocessor.preprocess(query)
        
        if not query_terms:
            return []
        
        # Get documents containing first term
        result_docs = set(self.index.index[query_terms[0]].keys()) if query_terms[0] in self.index.index else set()
        
        # Intersect with documents containing other terms
        for term in query_terms[1:]:
            if term in self.index.index:
                result_docs &= set(self.index.index[term].keys())
            else:
                return []  # If any term is not found, no documents match
        
        return list(result_docs)[:top_k]


def main():
    """Main execution function."""
    
    print("="*70)
    print("CISI INFORMATION RETRIEVAL SYSTEM - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # File paths (adjust these to your CISI dataset location)
    BASE_PATH = "./cisi_dataset/"  # Change this to your dataset path
    DOC_FILE = BASE_PATH + "CISI.ALL"
    QRY_FILE = BASE_PATH + "CISI.QRY"
    REL_FILE = BASE_PATH + "CISI.REL"
    
    # Step 1: Load dataset
    print("\n[1/6] Loading CISI dataset...")
    loader = CISIDataLoader()
    documents = loader.parse_cisi_documents(DOC_FILE)
    queries = loader.parse_cisi_queries(QRY_FILE)
    relevance_judgments = loader.parse_relevance_judgments(REL_FILE)
    
    print(f"  Loaded: {len(documents)} documents, {len(queries)} queries, "
          f"{len(relevance_judgments)} queries with relevance judgments")
    
    # Step 2: Build inverted index
    print("\n[2/6] Building inverted index...")
    index = InvertedIndex()
    index.build(documents)
    
    # Step 3: Initialize retrieval models
    print("\n[3/6] Initializing retrieval models...")
    models = RetrievalModels(index, len(documents))
    
    # Step 4: Run all models and collect results
    print("\n[4/6] Running retrieval models...")
    model_results = {}
    
    print("  Running TF-IDF model...")
    tfidf_results = {}
    for query_id, query_text in queries.items():
        tfidf_results[query_id] = models.tfidf_retrieval(query_text)
    model_results['TF-IDF'] = tfidf_results
    
    print("  Running Cosine Similarity model...")
    cosine_results = {}
    for query_id, query_text in queries.items():
        cosine_results[query_id] = models.cosine_similarity_retrieval(query_text)
    model_results['Cosine Similarity'] = cosine_results
    
    print("  Running Boolean AND model...")
    boolean_results = {}
    for query_id, query_text in queries.items():
        boolean_results[query_id] = models.boolean_and_retrieval(query_text)
    model_results['Boolean AND'] = boolean_results
    
    # Step 5: Comprehensive evaluation
    print("\n[5/6] Evaluating models with comprehensive metrics...")
    evaluator = IREvaluator(relevance_judgments)
    
    all_metrics = []
    for model_name, results in model_results.items():
        print(f"\n  Evaluating {model_name}...")
        metrics = evaluator.evaluate_model(results, model_name)
        all_metrics.append(metrics)
        evaluator.print_detailed_report(metrics)
    
    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    visualizer = IRVisualizer()
    visualizer.create_all_visualizations(all_metrics, "cisi_evaluation")
    
    # Save results to JSON file
    print("\nSaving evaluation results to JSON...")
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Results saved to: evaluation_results.json")
    
    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nModel Rankings (by MAP):")
    for rank, metrics in enumerate(all_metrics, 1):
        print(f"  {rank}. {metrics['model_name']}: MAP = {metrics['map']:.4f}")
    
    print("\nGenerated Files:")
    print("  - evaluation_results.json")
    print("  - cisi_evaluation_map_comparison.png")
    print("  - cisi_evaluation_precision_recall.png")
    print("  - cisi_evaluation_ndcg_comparison.png")
    print("  - cisi_evaluation_f1_scores.png")
    print("  - cisi_evaluation_comprehensive.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
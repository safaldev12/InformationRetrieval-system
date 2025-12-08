"""
Text Preprocessing Module
Handles tokenization, stopword removal, and stemming
"""

import re
from collections import Counter


class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'they', 'have', 'had', 'been',
            'what', 'when', 'where', 'who', 'which', 'why', 'how', 'this',
            'these', 'those', 'am', 'were', 'can', 'could', 'would', 'should'
        }
    
    def tokenize(self, text):
        """
        Convert text to lowercase and split into tokens.
        
        Args:
            text: Input text string
            
        Returns:
            list: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove common stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def stem(self, token):
        """
        Simple suffix-stripping stemmer.
        
        Args:
            token: Input token
            
        Returns:
            str: Stemmed token
        """
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'tion', 'ation', 'ness', 'ment']
        
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[:-len(suffix)]
        
        return token
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            list: Preprocessed tokens
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem tokens
        tokens = [self.stem(token) for token in tokens]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def get_term_frequency(self, tokens):
        """
        Calculate term frequency for a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Counter: Term frequencies
        """
        return Counter(tokens)


def preprocess_documents(documents, preprocessor):
    """
    Preprocess all documents in the collection.
    
    Args:
        documents: dict of {doc_id: text}
        preprocessor: TextPreprocessor instance
        
    Returns:
        dict: {doc_id: list of preprocessed tokens}
    """
    processed_docs = {}
    
    for doc_id, text in documents.items():
        processed_docs[doc_id] = preprocessor.preprocess(text)
    
    return processed_docs


def preprocess_queries(queries, preprocessor):
    """
    Preprocess all queries.
    
    Args:
        queries: dict of {query_id: text}
        preprocessor: TextPreprocessor instance
        
    Returns:
        dict: {query_id: list of preprocessed tokens}
    """
    processed_queries = {}
    
    for query_id, text in queries.items():
        processed_queries[query_id] = preprocessor.preprocess(text)
    
    return processed_queries
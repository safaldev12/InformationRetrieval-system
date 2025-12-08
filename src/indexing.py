"""
Indexing Module
Builds and manages inverted index structure
"""

from collections import defaultdict, Counter


class InvertedIndex:
    """Inverted index data structure for efficient document retrieval"""
    
    def __init__(self):
        # Index structure: {term: {doc_id: term_frequency}}
        self.index = defaultdict(lambda: defaultdict(int))
        
        # Document statistics
        self.doc_lengths = {}
        self.doc_vectors = {}
        self.num_docs = 0
        self.avg_doc_length = 0
        
        # Collection statistics
        self.total_terms = 0
        self.unique_terms = 0
    
    def build_index(self, processed_documents):
        """
        Build inverted index from preprocessed documents.
        
        Args:
            processed_documents: dict of {doc_id: list of tokens}
        """
        print("\n[Building Inverted Index]")
        print("-" * 60)
        
        self.num_docs = len(processed_documents)
        total_length = 0
        
        for doc_id, tokens in processed_documents.items():
            # Calculate document length
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
            
            # Calculate term frequencies
            term_counts = Counter(tokens)
            self.doc_vectors[doc_id] = term_counts
            
            # Build inverted index
            for term, count in term_counts.items():
                self.index[term][doc_id] = count
        
        # Calculate statistics
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        self.total_terms = total_length
        self.unique_terms = len(self.index)
        
        print(f"Indexed {self.num_docs} documents")
        print(f"Total terms: {self.total_terms:,}")
        print(f"Unique terms: {self.unique_terms:,}")
        print(f"Average document length: {self.avg_doc_length:.2f} tokens")
        print("-" * 60)
    
    def get_term_frequency(self, term, doc_id):
        """
        Get term frequency in a specific document.
        
        Args:
            term: Term to look up
            doc_id: Document ID
            
        Returns:
            int: Term frequency
        """
        return self.index[term].get(doc_id, 0)
    
    def get_document_frequency(self, term):
        """
        Get document frequency (number of documents containing term).
        
        Args:
            term: Term to look up
            
        Returns:
            int: Document frequency
        """
        return len(self.index[term])
    
    def get_posting_list(self, term):
        """
        Get complete posting list for a term.
        
        Args:
            term: Term to look up
            
        Returns:
            dict: {doc_id: term_frequency}
        """
        return dict(self.index[term])
    
    def get_document_length(self, doc_id):
        """
        Get length of a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            int: Document length
        """
        return self.doc_lengths.get(doc_id, 0)
    
    def get_document_vector(self, doc_id):
        """
        Get term frequency vector for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Counter: Term frequencies
        """
        return self.doc_vectors.get(doc_id, Counter())
    
    def get_statistics(self):
        """
        Get index statistics.
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            'num_documents': self.num_docs,
            'total_terms': self.total_terms,
            'unique_terms': self.unique_terms,
            'avg_doc_length': self.avg_doc_length
        }
    
    def print_sample_entries(self, num_terms=5):
        """
        Print sample index entries for verification.
        
        Args:
            num_terms: Number of sample terms to display
        """
        print("\n[Sample Index Entries]")
        print("-" * 60)
        
        sample_terms = list(self.index.keys())[:num_terms]
        
        for term in sample_terms:
            posting_list = self.get_posting_list(term)
            df = self.get_document_frequency(term)
            
            print(f"\nTerm: '{term}'")
            print(f"  Document Frequency: {df}")
            print(f"  Sample Postings: {dict(list(posting_list.items())[:5])}")
            if df > 5:
                print(f"  ... and {df - 5} more documents")
        
        print("-" * 60)
"""
Data Processing Module for CISI Dataset
Handles loading and parsing of CISI.ALL, CISI.QRY, CISI.REL
"""

import os


def parse_cisi_documents(filepath):
    """
    Parse CISI.ALL file to extract documents.
    
    Args:
        filepath: Path to CISI.ALL file
        
    Returns:
        dict: {doc_id: document_text}
    """
    documents = {}
    current_doc_id = None
    current_text = []
    current_field = None
    
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('.I '):
                # Save previous document
                if current_doc_id is not None and current_text:
                    documents[current_doc_id] = ' '.join(current_text)
                # Start new document
                current_doc_id = int(line.split()[1])
                current_text = []
                current_field = None
                
            elif line.startswith('.T'):
                current_field = 'T'  # Title
            elif line.startswith('.A'):
                current_field = 'A'  # Author
            elif line.startswith('.W'):
                current_field = 'W'  # Abstract/Content
            elif line.startswith('.X'):
                current_field = 'X'  # Cross-references (skip)
            elif current_field in ['T', 'W'] and line and not line.startswith('.'):
                # Collect title and abstract content
                current_text.append(line)
        
        # Save last document
        if current_doc_id is not None and current_text:
            documents[current_doc_id] = ' '.join(current_text)
    
    return documents


def parse_cisi_queries(filepath):
    """
    Parse CISI.QRY file to extract queries.
    
    Args:
        filepath: Path to CISI.QRY file
        
    Returns:
        dict: {query_id: query_text}
    """
    queries = {}
    current_query_id = None
    current_text = []
    current_field = None
    
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('.I '):
                # Save previous query
                if current_query_id is not None and current_text:
                    queries[current_query_id] = ' '.join(current_text)
                # Start new query
                current_query_id = int(line.split()[1])
                current_text = []
                current_field = None
                
            elif line.startswith('.W'):
                current_field = 'W'  # Query text
            elif current_field == 'W' and line and not line.startswith('.'):
                current_text.append(line)
        
        # Save last query
        if current_query_id is not None and current_text:
            queries[current_query_id] = ' '.join(current_text)
    
    return queries


def parse_cisi_relevance(filepath):
    """
    Parse CISI.REL file to extract relevance judgments.
    
    Args:
        filepath: Path to CISI.REL file
        
    Returns:
        dict: {query_id: set of relevant document_ids}
    """
    relevance = {}
    
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                
                if query_id not in relevance:
                    relevance[query_id] = set()
                relevance[query_id].add(doc_id)
    
    return relevance


def load_cisi_dataset(data_dir):
    """
    Load complete CISI dataset.
    
    Args:
        data_dir: Directory containing CISI files
        
    Returns:
        tuple: (documents, queries, relevance)
    """
    print("\n[Loading CISI Dataset from Kaggle]")
    print("-" * 60)
    
    # Load documents
    doc_path = os.path.join(data_dir, 'CISI.ALL')
    documents = parse_cisi_documents(doc_path)
    print(f"Loaded {len(documents)} documents from CISI.ALL")
    
    # Load queries
    query_path = os.path.join(data_dir, 'CISI.QRY')
    queries = parse_cisi_queries(query_path)
    print(f"Loaded {len(queries)} queries from CISI.QRY")
    
    # Load relevance judgments
    rel_path = os.path.join(data_dir, 'CISI.REL')
    relevance = parse_cisi_relevance(rel_path)
    print(f"Loaded relevance judgments for {len(relevance)} queries from CISI.REL")
    print("-" * 60)
    
    return documents, queries, relevance
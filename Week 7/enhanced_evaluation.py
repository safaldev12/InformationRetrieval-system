"""
Enhanced Evaluation Module for Information Retrieval Systems
Implements comprehensive evaluation metrics: Precision, Recall, MAP, nDCG, F1-score
Author: Safal Subedi
Course: TECH 400 - Introduction to Information Retrieval
"""

import math
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class IREvaluator:
    """
    Comprehensive evaluator for Information Retrieval systems.
    Computes multiple performance metrics for retrieval model comparison.
    """
    
    def __init__(self, relevance_judgments: Dict[int, Set[int]]):
        """
        Initialize evaluator with relevance judgments.
        
        Args:
            relevance_judgments: Dictionary mapping query_id to set of relevant doc_ids
        """
        self.relevance_judgments = relevance_judgments
        self.k_values = [5, 10, 20, 50]  # Standard cutoff values for evaluation
    
    def precision_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Precision@K: Proportion of relevant documents in top-K results.
        
        Formula: P@K = (Relevant docs in top-K) / K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            Precision value between 0 and 1
        """
        if k == 0 or len(retrieved_docs) == 0:
            return 0.0
        
        # Consider only top-K documents
        top_k = retrieved_docs[:k]
        
        # Count relevant documents in top-K
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_in_top_k / k
    
    def recall_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Recall@K: Proportion of relevant documents retrieved in top-K.
        
        Formula: R@K = (Relevant docs in top-K) / Total relevant docs
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            Recall value between 0 and 1
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        # Consider only top-K documents
        top_k = retrieved_docs[:k]
        
        # Count relevant documents in top-K
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def f1_score_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate F1-Score@K: Harmonic mean of Precision@K and Recall@K.
        
        Formula: F1@K = 2 * (P@K * R@K) / (P@K + R@K)
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            F1-score value between 0 and 1
        """
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, retrieved_docs: List[int], relevant_docs: Set[int]) -> float:
        """
        Calculate Average Precision (AP) for a single query.
        
        AP emphasizes retrieving relevant documents early in the ranking.
        
        Formula: AP = (1/R) * Σ(P(k) * rel(k))
        where R = total relevant docs, P(k) = precision at position k, 
        rel(k) = 1 if doc at k is relevant, 0 otherwise
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Average precision value between 0 and 1
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        num_relevant = 0
        sum_precisions = 0.0
        
        # Calculate precision at each position where a relevant document is found
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                num_relevant += 1
                # Precision at this position
                precision_at_rank = num_relevant / rank
                sum_precisions += precision_at_rank
        
        # Average over all relevant documents
        return sum_precisions / len(relevant_docs)
    
    def mean_average_precision(self, results: Dict[int, List[int]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across all queries.
        
        MAP is the mean of Average Precision scores for all queries.
        
        Args:
            results: Dictionary mapping query_id to list of retrieved doc_ids
            
        Returns:
            MAP value between 0 and 1
        """
        if len(results) == 0:
            return 0.0
        
        ap_scores = []
        
        for query_id, retrieved_docs in results.items():
            if query_id in self.relevance_judgments:
                relevant_docs = self.relevance_judgments[query_id]
                ap = self.average_precision(retrieved_docs, relevant_docs)
                ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    
    def dcg_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG) at position K.
        
        DCG measures the usefulness of documents based on their position in the result list.
        Documents appearing later are discounted.
        
        Formula: DCG@K = Σ(rel_i / log2(i + 1)) for i = 1 to K
        where rel_i = 1 if doc at position i is relevant, 0 otherwise
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            DCG value
        """
        dcg = 0.0
        
        for rank, doc_id in enumerate(retrieved_docs[:k], 1):
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance = 1 if doc_id in relevant_docs else 0
            
            # Discount by logarithm of rank position
            dcg += relevance / math.log2(rank + 1)
        
        return dcg
    
    def ideal_dcg_at_k(self, relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Ideal Discounted Cumulative Gain (IDCG) at position K.
        
        IDCG is the DCG of the ideal ranking where all relevant documents 
        appear at the top positions.
        
        Args:
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            IDCG value
        """
        # In ideal ranking, all relevant docs appear first
        num_relevant_in_top_k = min(len(relevant_docs), k)
        
        idcg = 0.0
        for rank in range(1, num_relevant_in_top_k + 1):
            # All top documents are relevant in ideal case
            idcg += 1.0 / math.log2(rank + 1)
        
        return idcg
    
    def ndcg_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG) at position K.
        
        nDCG normalizes DCG by the IDCG to get a value between 0 and 1.
        
        Formula: nDCG@K = DCG@K / IDCG@K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank position
            
        Returns:
            nDCG value between 0 and 1
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        dcg = self.dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg = self.ideal_dcg_at_k(relevant_docs, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_model(self, results: Dict[int, List[int]], model_name: str) -> Dict:
        """
        Comprehensive evaluation of a retrieval model.
        
        Calculates all metrics (Precision, Recall, F1, MAP, nDCG) at multiple cutoffs.
        
        Args:
            results: Dictionary mapping query_id to list of retrieved doc_ids
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'map': self.mean_average_precision(results),
            'precision': {},
            'recall': {},
            'f1_score': {},
            'ndcg': {}
        }
        
        # Calculate metrics at different K values
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            
            for query_id, retrieved_docs in results.items():
                if query_id in self.relevance_judgments:
                    relevant_docs = self.relevance_judgments[query_id]
                    
                    # Calculate all metrics at this K
                    precision_scores.append(
                        self.precision_at_k(retrieved_docs, relevant_docs, k)
                    )
                    recall_scores.append(
                        self.recall_at_k(retrieved_docs, relevant_docs, k)
                    )
                    f1_scores.append(
                        self.f1_score_at_k(retrieved_docs, relevant_docs, k)
                    )
                    ndcg_scores.append(
                        self.ndcg_at_k(retrieved_docs, relevant_docs, k)
                    )
            
            # Store average scores for this K
            metrics['precision'][f'P@{k}'] = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
            metrics['recall'][f'R@{k}'] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
            metrics['f1_score'][f'F1@{k}'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            metrics['ndcg'][f'nDCG@{k}'] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        
        return metrics
    
    def compare_models(self, model_results: Dict[str, Dict[int, List[int]]]) -> Dict:
        """
        Compare multiple retrieval models and rank them by performance.
        
        Args:
            model_results: Dictionary mapping model_name to results dictionary
            
        Returns:
            Dictionary containing comparison results for all models
        """
        comparison = {
            'models': [],
            'summary': {}
        }
        
        # Evaluate each model
        for model_name, results in model_results.items():
            metrics = self.evaluate_model(results, model_name)
            comparison['models'].append(metrics)
        
        # Rank models by MAP
        comparison['models'].sort(key=lambda x: x['map'], reverse=True)
        
        # Create summary
        comparison['summary']['best_model_by_map'] = comparison['models'][0]['model_name']
        comparison['summary']['best_map_score'] = comparison['models'][0]['map']
        
        return comparison
    
    def print_detailed_report(self, metrics: Dict):
        """
        Print a formatted evaluation report for a single model.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT: {metrics['model_name']}")
        print(f"{'='*70}\n")
        
        # MAP
        print(f"Mean Average Precision (MAP): {metrics['map']:.4f}\n")
        
        # Precision at K
        print("Precision @ K:")
        for k, score in metrics['precision'].items():
            print(f"  {k:8s}: {score:.4f}")
        
        # Recall at K
        print("\nRecall @ K:")
        for k, score in metrics['recall'].items():
            print(f"  {k:8s}: {score:.4f}")
        
        # F1-Score at K
        print("\nF1-Score @ K:")
        for k, score in metrics['f1_score'].items():
            print(f"  {k:8s}: {score:.4f}")
        
        # nDCG at K
        print("\nnDCG @ K:")
        for k, score in metrics['ndcg'].items():
            print(f"  {k:8s}: {score:.4f}")
        
        print(f"\n{'='*70}\n")


def demo_evaluation():
    """
    Demonstration of the evaluation system with sample data.
    """
    # Sample relevance judgments (query_id -> set of relevant doc_ids)
    relevance_judgments = {
        1: {10, 15, 23},
        2: {5, 12, 18, 25},
        3: {8, 14}
    }
    
    # Sample retrieval results (query_id -> ranked list of retrieved doc_ids)
    model_results = {
        'TF-IDF': {
            1: [10, 23, 7, 15, 20, 3, 11, 16, 8, 19],
            2: [12, 5, 18, 9, 25, 14, 22, 7, 11, 3],
            3: [14, 8, 20, 11, 5, 9, 16, 22, 7, 13]
        },
        'Cosine Similarity': {
            1: [23, 10, 7, 11, 15, 20, 16, 8, 3, 19],
            2: [5, 18, 12, 9, 14, 25, 7, 22, 11, 3],
            3: [8, 20, 14, 5, 11, 9, 22, 16, 7, 13]
        }
    }
    
    # Create evaluator
    evaluator = IREvaluator(relevance_judgments)
    
    # Evaluate and compare models
    comparison = evaluator.compare_models(model_results)
    
    # Print detailed reports
    for metrics in comparison['models']:
        evaluator.print_detailed_report(metrics)
    
    # Print comparison summary
    print("\nMODEL COMPARISON SUMMARY")
    print(f"Best Model: {comparison['summary']['best_model_by_map']}")
    print(f"Best MAP Score: {comparison['summary']['best_map_score']:.4f}")


if __name__ == "__main__":
    demo_evaluation()
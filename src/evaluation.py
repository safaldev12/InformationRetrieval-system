"""
Evaluation Module
Implements evaluation metrics: MAP, Recall@K, Average Precision
(Note: Precision@5 excluded as per instructor's guidance)
"""


def recall_at_k(retrieved, relevant, k):
    """
    Calculate Recall@K - proportion of relevant documents retrieved in top-K.
    
    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        float: Recall@K score
    """
    if not relevant or k == 0:
        return 0.0
    
    retrieved_k = set(retrieved[:k])
    relevant_retrieved = len(retrieved_k & relevant)
    
    return relevant_retrieved / len(relevant)


def average_precision(retrieved, relevant):
    """
    Calculate Average Precision for a single query.
    
    AP = (1/|Relevant|) * Σ(Precision@k * rel(k))
    where rel(k) = 1 if item at rank k is relevant, 0 otherwise
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        
    Returns:
        float: Average Precision score
    """
    if not relevant:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant)


def mean_average_precision(query_results, relevance_judgments):
    """
    Calculate Mean Average Precision (MAP) across all queries.
    
    MAP = (1/|Queries|) * Σ AP(q)
    
    Args:
        query_results: dict of {query_id: list of retrieved doc_ids}
        relevance_judgments: dict of {query_id: set of relevant doc_ids}
        
    Returns:
        float: MAP score
    """
    aps = []
    
    for query_id, retrieved in query_results.items():
        relevant = relevance_judgments.get(query_id, set())
        
        if relevant:  # Only calculate for queries with relevance judgments
            ap = average_precision(retrieved, relevant)
            aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0


def evaluate_model(query_results, relevance_judgments, k_values=[10, 20, 50]):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        query_results: dict of {query_id: list of retrieved doc_ids}
        relevance_judgments: dict of {query_id: set of relevant doc_ids}
        k_values: List of K values for Recall@K
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate MAP
    map_score = mean_average_precision(query_results, relevance_judgments)
    
    # Calculate Recall@K for different K values
    recall_scores = {}
    for k in k_values:
        recall_list = []
        for query_id, retrieved in query_results.items():
            relevant = relevance_judgments.get(query_id, set())
            if relevant:
                r_at_k = recall_at_k(retrieved, relevant, k)
                recall_list.append(r_at_k)
        
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
        recall_scores[f'Recall@{k}'] = avg_recall
    
    # Calculate number of queries evaluated
    num_queries = len([q for q in query_results.keys() 
                      if q in relevance_judgments and relevance_judgments[q]])
    
    return {
        'MAP': map_score,
        **recall_scores,
        'num_queries_evaluated': num_queries
    }


def print_evaluation_results(model_name, metrics):
    """
    Print evaluation results in a formatted way.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metric scores
    """
    print(f"\n[Evaluation Results: {model_name}]")
    print("-" * 60)
    print(f"Mean Average Precision (MAP): {metrics['MAP']:.4f}")
    
    for metric_name, value in metrics.items():
        if metric_name.startswith('Recall@'):
            print(f"{metric_name}: {value:.4f}")
    
    print(f"Queries Evaluated: {metrics['num_queries_evaluated']}")
    print("-" * 60)


def compare_models(model_results, relevance_judgments):
    """
    Compare multiple models and identify the best performer.
    
    Args:
        model_results: dict of {model_name: query_results}
        relevance_judgments: dict of relevance judgments
        
    Returns:
        dict: Comparison results
    """
    comparison = {}
    
    for model_name, query_results in model_results.items():
        metrics = evaluate_model(query_results, relevance_judgments)
        comparison[model_name] = metrics
    
    # Find best model by MAP
    best_model = max(comparison.items(), key=lambda x: x[1]['MAP'])
    
    return {
        'comparison': comparison,
        'best_model': best_model[0],
        'best_map': best_model[1]['MAP']
    }


def save_evaluation_results(comparison, output_file='results/evaluation.txt'):
    """
    Save evaluation results to a file.
    
    Args:
        comparison: Comparison results dictionary
        output_file: Output file path
    """
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("INFORMATION RETRIEVAL SYSTEM - EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for model_name, metrics in comparison['comparison'].items():
            f.write(f"{model_name}:\n")
            f.write(f"  MAP: {metrics['MAP']:.4f}\n")
            for metric_name, value in metrics.items():
                if metric_name.startswith('Recall@'):
                    f.write(f"  {metric_name}: {value:.4f}\n")
            f.write(f"  Queries Evaluated: {metrics['num_queries_evaluated']}\n")
            f.write("\n")
        
        f.write("-"*60 + "\n")
        f.write(f"Best Model: {comparison['best_model']} ")
        f.write(f"(MAP: {comparison['best_map']:.4f})\n")
        f.write("="*60 + "\n")
    
    print(f"\nEvaluation results saved to: {output_file}")


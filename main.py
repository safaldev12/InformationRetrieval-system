"""
Main Entry Point for CISI Information Retrieval System

Author: Safal
Course: TECH 400 - Introduction to Information Retrieval
Presidential Graduate School

This system implements multiple retrieval models on the CISI dataset:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Cosine Similarity
- Boolean AND

Evaluation Metrics:
- Mean Average Precision (MAP)
- Recall@10, Recall@20, Recall@50
"""

from src.data_processing import load_cisi_dataset
from src.preprocessing import TextPreprocessor, preprocess_documents, preprocess_queries
from src.indexing import InvertedIndex
from src.retrieval_models import search_all_queries
from src.evaluation import evaluate_model, print_evaluation_results, compare_models, save_evaluation_results


def main():
    """Main function to run the complete IR system"""
    
    print("\n" + "="*70)
    print(" "*15 + "CISI INFORMATION RETRIEVAL SYSTEM")
    print("="*70)
    print("\nAuthor: Safal")
    print("Course: TECH 400 - Presidential Graduate School")
    print("Dataset: CISI (from Kaggle)")
    print("\n" + "="*70)
    
    # Step 1: Load Dataset
    print("\n[STEP 1/6] Loading Dataset")
    print("="*70)
    
    try:
        documents, queries, relevance = load_cisi_dataset('data')
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not load CISI dataset: {e}")
        print("\nMake sure the following files exist in the 'data' folder:")
        print("  - CISI.ALL")
        print("  - CISI.QRY")
        print("  - CISI.REL")
        return
    
    # Step 2: Preprocessing
    print("\n[STEP 2/6] Text Preprocessing")
    print("="*70)
    
    preprocessor = TextPreprocessor()
    
    print("Preprocessing documents...")
    processed_documents = preprocess_documents(documents, preprocessor)
    print(f"Processed {len(processed_documents)} documents")
    
    print("Preprocessing queries...")
    processed_queries = preprocess_queries(queries, preprocessor)
    print(f"Processed {len(processed_queries)} queries")
    
    # Display preprocessing example
    sample_doc_id = list(documents.keys())[0]
    print(f"\nPreprocessing Example (Document {sample_doc_id}):")
    print(f"Original: {documents[sample_doc_id][:100]}...")
    print(f"Tokens: {processed_documents[sample_doc_id][:15]}")
    
    # Step 3: Build Index
    print("\n[STEP 3/6] Building Inverted Index")
    print("="*70)
    
    inverted_index = InvertedIndex()
    inverted_index.build_index(processed_documents)
    
    # Display sample index entries
    inverted_index.print_sample_entries(num_terms=5)
    
    # Step 4: Retrieval
    print("\n[STEP 4/6] Running Retrieval Models")
    print("="*70)
    
    models = ['tfidf', 'cosine', 'boolean']
    model_results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} model...")
        results = search_all_queries(processed_queries, inverted_index, model=model, top_k=100)
        model_results[model.upper()] = results
        print(f"  Completed! Retrieved results for {len(results)} queries")
    
    # Display sample search results
    print("\n[Sample Search Results]")
    print("-" * 60)
    sample_query_id = list(queries.keys())[0]
    print(f"Query {sample_query_id}: {queries[sample_query_id][:80]}...")
    print(f"\nTop 5 results (COSINE):")
    cosine_sample = model_results['COSINE'].get(sample_query_id, [])[:5]
    for rank, doc_id in enumerate(cosine_sample, 1):
        print(f"  {rank}. Document {doc_id}")
    
    # Step 5: Evaluation
    print("\n[STEP 5/6] Evaluating Models")
    print("="*70)
    
    for model_name, results in model_results.items():
        metrics = evaluate_model(results, relevance, k_values=[10, 20, 50])
        print_evaluation_results(model_name, metrics)
    
    # Step 6: Comparison and Results
    print("\n[STEP 6/6] Model Comparison")
    print("="*70)
    
    comparison = compare_models(model_results, relevance)
    
    print("\n[Overall Comparison]")
    print("-" * 60)
    print(f"{'Model':<15} {'MAP':<10} {'Recall@10':<12} {'Recall@20':<12}")
    print("-" * 60)
    
    for model_name in ['TFIDF', 'COSINE', 'BOOLEAN']:
        metrics = comparison['comparison'][model_name]
        print(f"{model_name:<15} {metrics['MAP']:<10.4f} "
              f"{metrics['Recall@10']:<12.4f} {metrics['Recall@20']:<12.4f}")
    
    print("-" * 60)
    print(f"\nBest Performing Model: {comparison['best_model']} ")
    print(f"with MAP of {comparison['best_map']:.4f}")
    
    # Save results
    print("\n[Saving Results]")
    print("-" * 60)
    save_evaluation_results(comparison, 'results/evaluation.txt')
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "SYSTEM SUMMARY")
    print("="*70)
    
    stats = inverted_index.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Documents: {stats['num_documents']:,}")
    print(f"  Queries: {len(queries)}")
    print(f"  Unique Terms: {stats['unique_terms']:,}")
    print(f"  Average Document Length: {stats['avg_doc_length']:.2f} tokens")
    
    print(f"\nModels Implemented:")
    print(f"  - TF-IDF (Term Frequency-Inverse Document Frequency)")
    print(f"  - Cosine Similarity")
    print(f"  - Boolean AND")
    
    print(f"\nEvaluation Metrics:")
    print(f"  - Mean Average Precision (MAP)")
    print(f"  - Recall@10")
    print(f"  - Recall@20")
    print(f"  - Recall@50")
    
    print(f"\nBest Model: {comparison['best_model']} (MAP: {comparison['best_map']:.4f})")
    
    print("\n" + "="*70)
    print(" "*20 + "SYSTEM EXECUTION COMPLETE!")
    print("="*70)
    
   


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
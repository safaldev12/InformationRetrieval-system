"""
Visualization Module for IR Evaluation Results
Generates comparison charts and graphs for retrieval model performance
Author: Safal Subedi
Course: TECH 400 - Introduction to Information Retrieval
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List
import numpy as np


class IRVisualizer:
    """
    Creates visualizations for Information Retrieval evaluation results.
    """
    
    def __init__(self, output_dir: str = './visualizations'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        self.markers = ['o', 's', '^', 'D', 'v']
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_map_comparison(self, metrics_list: List[Dict], save_path: str = None):
        """
        Create bar chart comparing MAP scores across models.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            save_path: Path to save the figure
        """
        model_names = [m['model_name'] for m in metrics_list]
        map_scores = [m['map'] for m in metrics_list]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(model_names, map_scores, color=self.colors[:len(model_names)], 
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Mean Average Precision (MAP)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Retrieval Models', fontsize=12, fontweight='bold')
        ax.set_title('MAP Comparison Across Retrieval Models', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(map_scores) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MAP comparison chart saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(self, metrics_list: List[Dict], save_path: str = None):
        """
        Create precision and recall curves across different K values.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        k_values = [5, 10, 20, 50]
        
        # Precision plot
        for idx, metrics in enumerate(metrics_list):
            model_name = metrics['model_name']
            precision_values = [metrics['precision'][f'P@{k}'] for k in k_values]
            
            ax1.plot(k_values, precision_values, marker=self.markers[idx], 
                    color=self.colors[idx], linewidth=2.5, markersize=8, 
                    label=model_name)
        
        ax1.set_xlabel('K (Number of Retrieved Documents)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Precision @ K', fontsize=11, fontweight='bold')
        ax1.set_title('Precision at Different K Values', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        
        # Recall plot
        for idx, metrics in enumerate(metrics_list):
            model_name = metrics['model_name']
            recall_values = [metrics['recall'][f'R@{k}'] for k in k_values]
            
            ax2.plot(k_values, recall_values, marker=self.markers[idx], 
                    color=self.colors[idx], linewidth=2.5, markersize=8, 
                    label=model_name)
        
        ax2.set_xlabel('K (Number of Retrieved Documents)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Recall @ K', fontsize=11, fontweight='bold')
        ax2.set_title('Recall at Different K Values', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to {save_path}")
        
        plt.close()
    
    def plot_ndcg_comparison(self, metrics_list: List[Dict], save_path: str = None):
        """
        Create grouped bar chart for nDCG scores at different K values.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            save_path: Path to save the figure
        """
        k_values = [5, 10, 20, 50]
        model_names = [m['model_name'] for m in metrics_list]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(k_values))
        width = 0.8 / len(model_names)
        
        for idx, metrics in enumerate(metrics_list):
            ndcg_values = [metrics['ndcg'][f'nDCG@{k}'] for k in k_values]
            offset = width * idx - (width * len(model_names) / 2) + width / 2
            
            bars = ax.bar(x + offset, ndcg_values, width, 
                         label=metrics['model_name'],
                         color=self.colors[idx], edgecolor='black', 
                         linewidth=0.8, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Only show label if value is significant
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('K Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('nDCG Score', fontsize=12, fontweight='bold')
        ax.set_title('nDCG Comparison Across Different K Values', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"nDCG comparison chart saved to {save_path}")
        
        plt.close()
    
    def plot_f1_scores(self, metrics_list: List[Dict], save_path: str = None):
        """
        Create line plot for F1-scores at different K values.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            save_path: Path to save the figure
        """
        k_values = [5, 10, 20, 50]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, metrics in enumerate(metrics_list):
            model_name = metrics['model_name']
            f1_values = [metrics['f1_score'][f'F1@{k}'] for k in k_values]
            
            ax.plot(k_values, f1_values, marker=self.markers[idx], 
                   color=self.colors[idx], linewidth=2.5, markersize=8, 
                   label=model_name)
        
        ax.set_xlabel('K (Number of Retrieved Documents)', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-Score @ K', fontsize=11, fontweight='bold')
        ax.set_title('F1-Score Comparison at Different K Values', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"F1-score chart saved to {save_path}")
        
        plt.close()
    
    def plot_comprehensive_comparison(self, metrics_list: List[Dict], save_path: str = None):
        """
        Create a comprehensive 2x2 grid of all main metrics.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        k_values = [5, 10, 20, 50]
        
        # 1. MAP Comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        model_names = [m['model_name'] for m in metrics_list]
        map_scores = [m['map'] for m in metrics_list]
        bars = ax1.bar(model_names, map_scores, color=self.colors[:len(model_names)], 
                       edgecolor='black', linewidth=1.2, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        ax1.set_ylabel('MAP Score', fontweight='bold')
        ax1.set_title('Mean Average Precision', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Precision @ K (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        for idx, metrics in enumerate(metrics_list):
            precision_values = [metrics['precision'][f'P@{k}'] for k in k_values]
            ax2.plot(k_values, precision_values, marker=self.markers[idx], 
                    color=self.colors[idx], linewidth=2, markersize=7, 
                    label=metrics['model_name'])
        ax2.set_xlabel('K Value', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('Precision at K', fontweight='bold', fontsize=12)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        
        # 3. Recall @ K (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        for idx, metrics in enumerate(metrics_list):
            recall_values = [metrics['recall'][f'R@{k}'] for k in k_values]
            ax3.plot(k_values, recall_values, marker=self.markers[idx], 
                    color=self.colors[idx], linewidth=2, markersize=7, 
                    label=metrics['model_name'])
        ax3.set_xlabel('K Value', fontweight='bold')
        ax3.set_ylabel('Recall', fontweight='bold')
        ax3.set_title('Recall at K', fontweight='bold', fontsize=12)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(k_values)
        
        # 4. nDCG @ K (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        for idx, metrics in enumerate(metrics_list):
            ndcg_values = [metrics['ndcg'][f'nDCG@{k}'] for k in k_values]
            ax4.plot(k_values, ndcg_values, marker=self.markers[idx], 
                    color=self.colors[idx], linewidth=2, markersize=7, 
                    label=metrics['model_name'])
        ax4.set_xlabel('K Value', fontweight='bold')
        ax4.set_ylabel('nDCG Score', fontweight='bold')
        ax4.set_title('Normalized Discounted Cumulative Gain', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(k_values)
        
        fig.suptitle('Comprehensive IR Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive comparison chart saved to {save_path}")
        
        plt.close()
    
    def create_all_visualizations(self, metrics_list: List[Dict], prefix: str = "ir_eval"):
        """
        Generate all visualization charts.
        
        Args:
            metrics_list: List of metric dictionaries from evaluator
            prefix: Prefix for output filenames
        """
        print("\nGenerating visualizations...")
        print("-" * 50)
        
        self.plot_map_comparison(metrics_list, f"{prefix}_map_comparison.png")
        self.plot_precision_recall_curves(metrics_list, f"{prefix}_precision_recall.png")
        self.plot_ndcg_comparison(metrics_list, f"{prefix}_ndcg_comparison.png")
        self.plot_f1_scores(metrics_list, f"{prefix}_f1_scores.png")
        self.plot_comprehensive_comparison(metrics_list, f"{prefix}_comprehensive.png")
        
        print("-" * 50)
        print("All visualizations generated successfully!")


if __name__ == "__main__":
    # Demo visualization with sample data
    sample_metrics = [
        {
            'model_name': 'TF-IDF',
            'map': 0.1494,
            'precision': {'P@5': 0.142, 'P@10': 0.120, 'P@20': 0.095, 'P@50': 0.068},
            'recall': {'R@5': 0.080, 'R@10': 0.120, 'R@20': 0.168, 'R@50': 0.284},
            'f1_score': {'F1@5': 0.101, 'F1@10': 0.120, 'F1@20': 0.119, 'F1@50': 0.110},
            'ndcg': {'nDCG@5': 0.152, 'nDCG@10': 0.145, 'nDCG@20': 0.140, 'nDCG@50': 0.168}
        },
        {
            'model_name': 'Cosine Similarity',
            'map': 0.1071,
            'precision': {'P@5': 0.110, 'P@10': 0.090, 'P@20': 0.082, 'P@50': 0.056},
            'recall': {'R@5': 0.062, 'R@10': 0.090, 'R@20': 0.147, 'R@50': 0.233},
            'f1_score': {'F1@5': 0.078, 'F1@10': 0.090, 'F1@20': 0.103, 'F1@50': 0.090},
            'ndcg': {'nDCG@5': 0.118, 'nDCG@10': 0.110, 'nDCG@20': 0.122, 'nDCG@50': 0.142}
        },
        {
            'model_name': 'Boolean AND',
            'map': 0.0005,
            'precision': {'P@5': 0.001, 'P@10': 0.001, 'P@20': 0.001, 'P@50': 0.001},
            'recall': {'R@5': 0.001, 'R@10': 0.001, 'R@20': 0.001, 'R@50': 0.001},
            'f1_score': {'F1@5': 0.001, 'F1@10': 0.001, 'F1@20': 0.001, 'F1@50': 0.001},
            'ndcg': {'nDCG@5': 0.001, 'nDCG@10': 0.001, 'nDCG@20': 0.001, 'nDCG@50': 0.002}
        }
    ]
    
    visualizer = IRVisualizer()
    visualizer.create_all_visualizations(sample_metrics, "demo")
    print("\nDemo visualizations created!")
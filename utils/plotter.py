# utils/plotter.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MetricsPlotter:
    def __init__(self, save_dir='results/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style('whitegrid')
    
    def plot_all(self, metrics):
        if not metrics['rounds']:
            return
        self._plot_main_metrics(metrics)
        self._plot_classification_metrics(metrics)
        if metrics.get('per_node'):
            self._plot_per_node(metrics)
        self._plot_combined(metrics)
    
    def _plot_main_metrics(self, metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(metrics['rounds'], metrics['loss'], 'o-', linewidth=2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(metrics['rounds'], metrics['accuracy'], 'o-', linewidth=2, color='green')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_metrics(self, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(metrics['rounds'], metrics['precision'], 'o-', label='Precision', linewidth=2)
        ax.plot(metrics['rounds'], metrics['recall'], 's-', label='Recall', linewidth=2)
        ax.plot(metrics['rounds'], metrics['f1'], '^-', label='F1-Score', linewidth=2)
        ax.plot(metrics['rounds'], metrics['auc'], 'd-', label='AUC', linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_node(self, metrics):
        node_count = len(metrics['per_node'])
        if node_count == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            for node_id, node_metrics in metrics['per_node'].items():
                if metric in node_metrics and node_metrics[metric]:
                    ax.plot(metrics['rounds'], node_metrics[metric], 
                           'o-', label=f'Node {node_id}', alpha=0.7)
            
            ax.set_xlabel('Round')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} per Node')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_node_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined(self, metrics):
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics['rounds'], metrics['loss'], 'o-', linewidth=2, color='red')
        ax1.set_title('Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics['rounds'], metrics['accuracy'], 'o-', linewidth=2, color='green')
        ax2.set_title('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(metrics['rounds'], metrics['precision'], 'o-', label='Precision')
        ax3.plot(metrics['rounds'], metrics['recall'], 's-', label='Recall')
        ax3.set_title('Precision & Recall')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(metrics['rounds'], metrics['f1'], '^-', label='F1-Score')
        ax4.plot(metrics['rounds'], metrics['auc'], 'd-', label='AUC')
        ax4.set_title('F1-Score & AUC')
        ax4.set_ylim([0, 1])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        final_metrics = [
            ['Loss', f"{metrics['loss'][-1]:.4f}"],
            ['Accuracy', f"{metrics['accuracy'][-1]:.4f}"],
            ['Precision', f"{metrics['precision'][-1]:.4f}"],
            ['Recall', f"{metrics['recall'][-1]:.4f}"],
            ['F1-Score', f"{metrics['f1'][-1]:.4f}"],
            ['AUC', f"{metrics['auc'][-1]:.4f}"]
        ]
        
        table = ax5.table(cellText=final_metrics,
                         colLabels=['Metric', 'Final Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.3, 0.2, 0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.savefig(self.save_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
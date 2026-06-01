"""
VisualizationManager - Generates and saves plots for model analysis.
Implements confusion matrix, ROC/PR curves, automata state diagrams, and more.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

from src.core.config_manager import ConfigurationManager


class VisualizationManager:
    """
    Generates and saves visualization plots for model analysis.
    Supports confusion matrix, ROC/PR curves, automata state diagrams,
    transition probability heatmaps, and parameter sensitivity plots.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save figures
        """
        self.config = ConfigurationManager()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Figure format settings
        self.figure_formats = self.config.get(
            "visualization.figure_formats", ["png", "pdf"]
        )
        self.figure_dpi = self.config.get("visualization.dpi", 150)
        self.figure_size = self.config.get("visualization.figure_size", (10, 8))

    def _save_figure(self, fig: Figure, filename: str) -> List[str]:
        """
        Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename (without extension)
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for fmt in self.figure_formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, dpi=self.figure_dpi, bbox_inches="tight")
            saved_paths.append(filepath)
        
        return saved_paths

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        model_name: str,
        dataset_name: str,
        labels: List[str] = ["Normal", "Anomaly"],
    ) -> str:
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name of the model
            dataset_name: Name of the dataset
            labels: Label names for the confusion matrix
            
        Returns:
            Path to the saved figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        
        # Add labels and title
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name} on {dataset_name}")
        
        # Save figure
        filename = f"confusion_matrix_{model_name}_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def plot_roc_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        model_name: str,
        dataset_name: str,
        label: Optional[str] = None,
    ) -> str:
        """
        Generate and save ROC curve plot.
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores (probabilities or confidence)
            model_name: Name of the model
            dataset_name: Name of the dataset
            label: Optional label for the curve
            
        Returns:
            Path to the saved figure
        """
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot ROC curve
        curve_label = label if label else model_name
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"{curve_label} (AUC = {roc_auc:.3f})",
        )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        
        # Configure plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name} on {dataset_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"roc_curve_{model_name}_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def plot_pr_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        model_name: str,
        dataset_name: str,
        label: Optional[str] = None,
    ) -> str:
        """
        Generate and save Precision-Recall curve plot.
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores
            model_name: Name of the model
            dataset_name: Name of the dataset
            label: Optional label for the curve
            
        Returns:
            Path to the saved figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot PR curve
        curve_label = label if label else model_name
        ax.plot(
            recall,
            precision,
            color="darkblue",
            lw=2,
            label=f"{curve_label} (AP = {ap_score:.3f})",
        )
        
        # Configure plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_name} on {dataset_name}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"pr_curve_{model_name}_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def plot_multiple_curves(
        self,
        y_true: List[int],
        y_scores_dict: Dict[str, List[float]],
        plot_type: str = "roc",
        dataset_name: str = "unknown",
    ) -> str:
        """
        Generate plot with multiple curves for comparison.
        
        Args:
            y_true: Ground truth labels
            y_scores_dict: Dictionary mapping model_name -> y_scores
            plot_type: Type of plot ("roc" or "pr")
            dataset_name: Name of the dataset
            
        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(y_scores_dict)))
        
        for (model_name, y_scores), color in zip(y_scores_dict.items(), colors):
            if plot_type == "roc":
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                score = auc(fpr, tpr)
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    lw=2,
                    label=f"{model_name} (AUC = {score:.3f})",
                )
            elif plot_type == "pr":
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                score = average_precision_score(y_true, y_scores)
                ax.plot(
                    recall,
                    precision,
                    color=color,
                    lw=2,
                    label=f"{model_name} (AP = {score:.3f})",
                )
        
        # Configure plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate" if plot_type == "roc" else "Recall")
        ax.set_ylabel("True Positive Rate" if plot_type == "roc" else "Precision")
        ax.set_title(f"Multiple {plot_type.upper()} Curves - {dataset_name}")
        ax.legend(loc="lower right" if plot_type == "roc" else "lower left")
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"multiple_{plot_type}_curves_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def plot_state_diagram(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        vocabulary: List[str],
        model_name: str,
        dataset_name: str,
        top_n: Optional[int] = None,
    ) -> str:
        """
        Generate and save automata state diagram using Graphviz.
        
        Args:
            transition_matrix: Dictionary mapping current state to next state probabilities
            vocabulary: List of known patterns
            model_name: Name of the model
            dataset_name: Name of the dataset
            top_n: Optional number of most probable transitions to show
            
        Returns:
            Path to the saved figure or empty string if Graphviz not available
        """
        if not GRAPHVIZ_AVAILABLE:
            print("Graphviz is not installed. Skipping state diagram generation.")
            return ""
        
        try:
            # Create graph
            dot = graphviz.Digraph(
                f"automata_{model_name}_{dataset_name}",
                format="png",
            )
            
            # Add nodes
            for pattern in vocabulary:
                dot.node(pattern, pattern)
            
            # Add edges with probabilities
            for current, next_states in transition_matrix.items():
                # Sort by probability and optionally limit
                sorted_transitions = sorted(
                    next_states.items(), key=lambda x: x[1], reverse=True
                )
                
                if top_n:
                    sorted_transitions = sorted_transitions[:top_n]
                
                for next_state, prob in sorted_transitions:
                    dot.edge(current, next_state, label=f"{prob:.3f}")
            
            # Save figure
            filepath = os.path.join(self.output_dir, f"state_diagram_{model_name}_{dataset_name}")
            dot.render(filepath, view=False, cleanup=True)
            
            return f"{filepath}.png"
        except Exception as e:
            print(f"Graphviz state diagram generation failed: {e}")
            return ""

    def plot_transition_heatmap(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        vocabulary: List[str],
        model_name: str,
        dataset_name: str,
        top_n: Optional[int] = None,
    ) -> str:
        """
        Generate and save transition probability heatmap.
        
        Args:
            transition_matrix: Dictionary mapping current state to next state probabilities
            vocabulary: List of known patterns
            model_name: Name of the model
            dataset_name: Name of the dataset
            top_n: Optional number of most probable transitions to show
            
        Returns:
            Path to the saved figure
        """
        # Create probability matrix
        if top_n:
            # Get top transitions for each state
            top_patterns = set()
            for current, next_states in transition_matrix.items():
                sorted_transitions = sorted(
                    next_states.items(), key=lambda x: x[1], reverse=True
                )[:top_n]
                top_patterns.add(current)
                for next_state, _ in sorted_transitions:
                    top_patterns.add(next_state)
            
            # Filter vocabulary
            filtered_vocab = [p for p in vocabulary if p in top_patterns]
        else:
            filtered_vocab = vocabulary
        
        # Create matrix
        n = len(filtered_vocab)
        if n == 0:
            print("No transitions to plot")
            return ""
        
        prob_matrix = np.zeros((n, n))
        for i, current in enumerate(filtered_vocab):
            if current in transition_matrix:
                for j, next_state in enumerate(filtered_vocab):
                    if next_state in transition_matrix[current]:
                        prob_matrix[i, j] = transition_matrix[current][next_state]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            prob_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=filtered_vocab,
            yticklabels=filtered_vocab,
            ax=ax,
            cbar_kws={"label": "Transition Probability"},
        )
        
        # Configure plot
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")
        ax.set_title(f"Transition Probability Heatmap - {model_name} on {dataset_name}")
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        # Save figure
        filename = f"transition_heatmap_{model_name}_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def plot_sensitivity_analysis(
        self,
        sensitivity_results: List[Dict[str, Any]],
        model_name: str = "automata",
        dataset_name: str = "unknown",
    ) -> List[str]:
        """
        Generate parameter sensitivity plots.
        
        Args:
            sensitivity_results: List of sensitivity analysis results
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            List of saved figure paths
        """
        if not sensitivity_results:
            print("No sensitivity results to plot")
            return []
        
        # Extract parameters and metrics
        window_sizes = set()
        alphabet_sizes = set()
        
        for result in sensitivity_results:
            if "window_size" in result:
                window_sizes.add(result["window_size"])
            if "alphabet_size" in result:
                alphabet_sizes.add(result["alphabet_size"])
        
        if not window_sizes or not alphabet_sizes:
            print("Missing window_size or alphabet_size in results")
            return []
        
        # Create 2D grids for each metric
        metrics = ["accuracy", "precision", "recall", "f1"]
        saved_paths = []
        
        for metric in metrics:
            # Create grid
            ws_list = sorted(list(window_sizes))
            ab_list = sorted(list(alphabet_sizes))
            
            grid = np.zeros((len(ab_list), len(ws_list)))
            
            for result in sensitivity_results:
                ws = result.get("window_size")
                ab = result.get("alphabet_size")
                
                if ws is None or ab is None:
                    continue
                
                if metric in result.get("metrics", {}):
                    i = ab_list.index(ab)
                    j = ws_list.index(ws)
                    grid[i, j] = result["metrics"][metric]
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot heatmap
            sns.heatmap(
                grid,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                xticklabels=ws_list,
                yticklabels=ab_list,
                ax=ax,
                cbar_kws={"label": metric.capitalize()},
            )
            
            # Configure plot
            ax.set_xlabel("Window Size")
            ax.set_ylabel("Alphabet Size")
            ax.set_title(f"{metric.capitalize()} Sensitivity - {model_name} on {dataset_name}")
            
            # Save figure
            filename = f"sensitivity_{metric}_{model_name}_{dataset_name}"
            paths = self._save_figure(fig, filename)
            saved_paths.extend(paths)
            
            plt.close(fig)
        
        return saved_paths

    def plot_summary_across_seeds(
        self,
        metrics_by_seed: Dict[int, Dict[str, float]],
        metric_name: str,
        model_name: str,
        dataset_name: str,
    ) -> str:
        """
        Generate summary plot across multiple seeds.
        
        Args:
            metrics_by_seed: Dictionary mapping seed -> metrics dict
            metric_name: Name of the metric to plot
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Path to the saved figure
        """
        seeds = sorted(metrics_by_seed.keys())
        values = [metrics_by_seed[seed].get(metric_name, 0) for seed in seeds]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot individual seed results
        ax.plot(seeds, values, "o-", label=f"{metric_name} per seed")
        
        # Add mean and std
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean ({mean_val:.4f} ± {std_val:.4f})")
        ax.fill_between(
            seeds,
            mean_val - std_val,
            mean_val + std_val,
            alpha=0.2,
            color="red",
            label="±1 std",
        )
        
        # Configure plot
        ax.set_xlabel("Seed")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} Across Seeds - {model_name} on {dataset_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"summary_{metric_name}_across_seeds_{model_name}_{dataset_name}"
        saved_paths = self._save_figure(fig, filename)
        
        plt.close(fig)
        
        return saved_paths[0] if saved_paths else ""

    def generate_all_reports(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float],
        transition_matrix: Dict[str, Dict[str, float]],
        vocabulary: List[str],
        model_name: str,
        dataset_name: str,
    ) -> Dict[str, str]:
        """
        Generate all available reports for a model.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Prediction scores
            transition_matrix: Automata transition matrix
            vocabulary: Automata vocabulary
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping report type to saved path
        """
        reports = {}
        
        # Confusion matrix
        reports["confusion_matrix"] = self.plot_confusion_matrix(
            y_true, y_pred, model_name, dataset_name
        )
        
        # ROC curve
        reports["roc_curve"] = self.plot_roc_curve(
            y_true, y_scores, model_name, dataset_name
        )
        
        # PR curve
        reports["pr_curve"] = self.plot_pr_curve(
            y_true, y_scores, model_name, dataset_name
        )
        
        # State diagram (if Graphviz available)
        if GRAPHVIZ_AVAILABLE:
            reports["state_diagram"] = self.plot_state_diagram(
                transition_matrix, vocabulary, model_name, dataset_name
            )
        
        # Transition heatmap
        reports["transition_heatmap"] = self.plot_transition_heatmap(
            transition_matrix, vocabulary, model_name, dataset_name
        )
        
        return reports

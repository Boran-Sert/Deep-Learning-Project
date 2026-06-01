"""
ReportManager - Observer pattern implementation for event-driven reporting.
Aggregates metrics across multiple seeds and folds, computes statistics.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from src.orchestration.event_bus import EventBus
from src.orchestration.events import (
    ModelTrainedEvent,
    EvaluationCompletedEvent,
    AutomataDecisionEvent,
    SensitivityAnalysisEvent,
)
from src.core.config_manager import ConfigurationManager


@dataclass
class MetricStats:
    """Statistics for a single metric across multiple runs."""
    mean: float = 0.0
    std: float = 0.0
    values: List[float] = field(default_factory=list)


class ReportManager:
    """
    Observer component that listens to events and computes statistics.
    Aggregates metrics across seeds and folds for Table 5 format reporting.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.config = ConfigurationManager()
        self.event_bus = event_bus or EventBus()
        
        # Subscribe to all relevant events
        self.event_bus.subscribe(ModelTrainedEvent, self._on_model_trained)
        self.event_bus.subscribe(EvaluationCompletedEvent, self._on_evaluation_completed)
        self.event_bus.subscribe(AutomataDecisionEvent, self._on_automata_decision)
        self.event_bus.subscribe(SensitivityAnalysisEvent, self._on_sensitivity_analysis)
        
        # Storage for metrics
        self.training_times: Dict[str, List[float]] = defaultdict(list)
        self.inference_times: Dict[str, List[float]] = defaultdict(list)
        self.metrics_by_model: Dict[str, Dict[str, MetricStats]] = defaultdict(
            lambda: defaultdict(MetricStats)
        )
        self.automata_decisions: List[Dict[str, Any]] = []
        self.sensitivity_results: List[Dict[str, Any]] = []
        
        # Track unique models and seeds
        self.models_seen: set = set()
        self.seeds_seen: set = set()
        self.folds_seen: set = set()

    def _on_model_trained(self, event: ModelTrainedEvent) -> None:
        """Handle ModelTrainedEvent - record training duration."""
        key = f"{event.model_name}_{event.dataset_name}"
        # Training duration is set by the orchestrator via event
        if event.training_duration_seconds > 0:
            self.training_times[key].append(event.training_duration_seconds)

    def _on_evaluation_completed(self, event: EvaluationCompletedEvent) -> None:
        """Handle EvaluationCompletedEvent - record metrics."""
        model_key = f"{event.model_name}_{event.train_dataset}"
        
        # Extract metrics
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            if metric_name in event.metrics:
                value = event.metrics[metric_name]
                self.metrics_by_model[model_key][metric_name].values.append(value)
        
        # Track unique identifiers
        self.models_seen.add(event.model_name)
        self.seeds_seen.add(event.seed)
        if event.fold_idx is not None:
            self.folds_seen.add(event.fold_idx)

    def _on_automata_decision(self, event: AutomataDecisionEvent) -> None:
        """Handle AutomataDecisionEvent - record automata decision details."""
        decision_record = {
            "scenario": event.scenario,
            "unseen_rate": event.unseen_rate,
            "metrics": event.metrics,
            "fold_idx": event.fold_idx,
            "mapping_log": event.mapping_log,
        }
        self.automata_decisions.append(decision_record)

    def _on_sensitivity_analysis(self, event: SensitivityAnalysisEvent) -> None:
        """Handle SensitivityAnalysisEvent - store sensitivity results."""
        self.sensitivity_results.extend(event.results)

    def compute_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute mean and std for all metrics across all seeds/folds.
        Returns: Dict[model_key, Dict[metric_name, Dict["mean", "std"]]]
        """
        stats = {}
        
        for model_key, metrics in self.metrics_by_model.items():
            stats[model_key] = {}
            for metric_name, metric_stats in metrics.items():
                if metric_stats.values:
                    stats[model_key][metric_name] = {
                        "mean": float(np.mean(metric_stats.values)),
                        "std": float(np.std(metric_stats.values)),
                    }
        
        return stats

    def get_table5_format(self) -> List[Dict[str, Any]]:
        """
        Format metrics in Table 5 style: Model, Seed, Fold, F1, Accuracy, Precision, Recall, Training Time, Inference Time.
        """
        table_rows = []
        
        # Group metrics by model, seed, fold
        metrics_by_group: Dict[tuple, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        for model_key, metrics in self.metrics_by_model.items():
            # Parse model_key to extract model and dataset
            parts = model_key.rsplit("_", 1)
            model_name = parts[0] if len(parts) > 1 else model_key
            dataset_name = parts[1] if len(parts) > 1 else "unknown"
            
            # We need to track which seed/fold each metric belongs to
            # This requires re-processing the events - simplified approach here
            for metric_name, metric_stats in metrics.items():
                for value in metric_stats.values:
                    # This is a simplified representation
                    # In practice, we'd track seed/fold per metric value
                    pass
        
        # For now, return aggregated stats
        stats = self.compute_statistics()
        for model_key, metrics in stats.items():
            parts = model_key.rsplit("_", 1)
            model_name = parts[0] if len(parts) > 1 else model_name
            dataset_name = parts[1] if len(parts) > 1 else "unknown"
            
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Seeds": len(self.seeds_seen),
                "Folds": len(self.folds_seen) if self.folds_seen else 1,
            }
            
            for metric_name in ["accuracy", "precision", "recall", "f1"]:
                if metric_name in metrics:
                    row[f"{metric_name}_mean"] = metrics[metric_name]["mean"]
                    row[f"{metric_name}_std"] = metrics[metric_name]["std"]
            
            # Add training time if available
            if model_key in self.training_times:
                times = self.training_times[model_key]
                row["Training_Time_mean"] = float(np.mean(times))
                row["Training_Time_std"] = float(np.std(times))
            
            table_rows.append(row)
        
        return table_rows

    def save_report(self, output_dir: str = "reports") -> str:
        """
        Save all reports to JSON files.
        Returns path to the saved report.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute statistics
        stats = self.compute_statistics()
        table5 = self.get_table5_format()
        
        report = {
            "summary": {
                "models_seen": list(self.models_seen),
                "seeds_seen": list(self.seeds_seen),
                "folds_seen": list(self.folds_seen),
            },
            "metrics_statistics": stats,
            "table5_format": table5,
            "training_times": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "count": len(v),
                }
                for k, v in self.training_times.items()
            },
            "automata_decisions": self.automata_decisions,
            "sensitivity_results": self.sensitivity_results,
        }
        
        report_path = os.path.join(output_dir, "experiment_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return report_path

    def print_summary(self) -> None:
        """Print a human-readable summary of the experiment results."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY REPORT")
        print("=" * 60)
        
        stats = self.compute_statistics()
        for model_key, metrics in stats.items():
            print(f"\nModel: {model_key}")
            print("-" * 40)
            for metric_name, values in metrics.items():
                print(
                    f"  {metric_name.upper():12s}: {values['mean']:.4f} ± {values['std']:.4f}"
                )
        
        if self.training_times:
            print("\nTraining Times:")
            print("-" * 40)
            for model_key, times in self.training_times.items():
                print(
                    f"  {model_key:30s}: {np.mean(times):.2f}s ± {np.std(times):.2f}s"
                )
        
        print("\n" + "=" * 60)

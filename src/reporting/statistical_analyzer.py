"""
StatisticalAnalyzer - Statistical significance testing for model comparisons.
Implements Wilcoxon signed-rank test and McNemar test.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


class StatisticalAnalyzer:
    """
    Performs statistical significance tests on model performance metrics.
    Supports Wilcoxon signed-rank test for paired samples and McNemar test for proportions.
    """

    def __init__(self):
        self.alpha = 0.05  # Significance level

    def wilcoxon_test(
        self, 
        model_a_metrics: List[float], 
        model_b_metrics: List[float]
    ) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Args:
            model_a_metrics: List of metric values for model A
            model_b_metrics: List of metric values for model B
            
        Returns:
            Dictionary with test statistic, p-value, and significance indication
        """
        if len(model_a_metrics) != len(model_b_metrics):
            raise ValueError("Both samples must have the same size for Wilcoxon test")
        
        if len(model_a_metrics) < 2:
            raise ValueError("At least 2 paired samples are required")
        
        stat, p_value = stats.wilcoxon(model_a_metrics, model_b_metrics)
        
        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": str(p_value < self.alpha),
            "alpha": self.alpha,
        }

    def mcnemar_test(
        self, 
        model_a_predictions: List[int], 
        model_b_predictions: List[int], 
        ground_truth: List[int]
    ) -> Dict[str, float]:
        """
        Perform McNemar test for comparing paired proportions (classification accuracy).
        
        Args:
            model_a_predictions: Binary predictions from model A
            model_b_predictions: Binary predictions from model B
            ground_truth: Ground truth binary labels
            
        Returns:
            Dictionary with test statistic, p-value, and significance indication
        """
        if len(model_a_predictions) != len(model_b_predictions) or \
           len(model_a_predictions) != len(ground_truth):
            raise ValueError("All inputs must have the same length")
        
        # Build contingency table
        # a: both correct, b: A correct B wrong, c: A wrong B correct, d: both wrong
        a = b = c = d = 0
        
        for pred_a, pred_b, truth in zip(model_a_predictions, model_b_predictions, ground_truth):
            correct_a = (pred_a == truth)
            correct_b = (pred_b == truth)
            
            if correct_a and correct_b:
                a += 1
            elif correct_a and not correct_b:
                b += 1
            elif not correct_a and correct_b:
                c += 1
            else:
                d += 1
        
        # McNemar test statistic (with continuity correction)
        n = b + c
        if n == 0:
            return {
                "test": "mcnemar",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": "False",
                "alpha": self.alpha,
                "contingency_table": {"a": a, "b": b, "c": c, "d": d},
            }
        
        # Continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
        p_value = 1 - stats.chi2.cdf(stat, 1)
        
        return {
            "test": "mcnemar",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": str(p_value < self.alpha),
            "alpha": self.alpha,
            "contingency_table": {"a": a, "b": b, "c": c, "d": d},
        }

    def compare_models(
        self,
        model_a_name: str,
        model_b_name: str,
        model_a_metrics: Dict[str, List[float]],
        model_b_metrics: Dict[str, List[float]],
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two models across multiple metrics using appropriate statistical tests.
        
        Args:
            model_a_name: Name of model A
            model_b_name: Name of model B
            model_a_metrics: Dict of metric_name -> list of values for model A
            model_b_metrics: Dict of metric_name -> list of values for model B
            metric_names: Optional list of metrics to compare (default: all common metrics)
            
        Returns:
            Dict of metric_name -> statistical test results
        """
        if metric_names is None:
            metric_names = list(set(model_a_metrics.keys()) & set(model_b_metrics.keys()))
        
        results = {}
        for metric_name in metric_names:
            if metric_name not in model_a_metrics or metric_name not in model_b_metrics:
                continue
            
            a_values = model_a_metrics[metric_name]
            b_values = model_b_metrics[metric_name]
            
            # Use Wilcoxon for continuous metrics
            results[metric_name] = self.wilcoxon_test(a_values, b_values)
        
        return results

    def apply_multiple_testing_correction(
        self, 
        p_values: List[float], 
        method: str = "bonferroni"
    ) -> List[float]:
        """
        Apply multiple testing correction to p-values.
        
        Args:
            p_values: List of raw p-values
            method: Correction method ("bonferroni" or "holm")
            
        Returns:
            List of corrected p-values
        """
        n = len(p_values)
        corrected = []
        
        if method == "bonferroni":
            corrected = [min(p * n, 1.0) for p in p_values]
        elif method == "holm":
            sorted_indices = np.argsort(p_values)
            corrected_p = [0.0] * n
            max_p = 0.0
            
            for i, idx in enumerate(sorted_indices):
                rank = n - i
                corrected_p[idx] = min(p_values[idx] * rank, 1.0, max_p)
                max_p = corrected_p[idx]
            
            corrected = corrected_p
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return corrected

    def format_test_results(
        self, 
        results: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Format statistical test results as a human-readable string.
        
        Args:
            results: Dictionary from compare_models or single test
            
        Returns:
            Formatted string with test results
        """
        lines = []
        lines.append("Statistical Test Results")
        lines.append("=" * 50)
        
        for metric_name, test_result in results.items():
            lines.append(f"\n{metric_name.upper()}:")
            lines.append(f"  Test: {test_result['test']}")
            lines.append(f"  Statistic: {test_result['statistic']:.4f}")
            lines.append(f"  P-value: {test_result['p_value']:.4f}")
            significant = str(test_result['significant'])
            lines.append(
                f"  Significant at α={test_result['alpha']}: {'Yes' if significant == 'True' else 'No'}"
            )
        
        return "\n".join(lines)


def convert_to_serializable(obj):
    """
    Convert numpy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

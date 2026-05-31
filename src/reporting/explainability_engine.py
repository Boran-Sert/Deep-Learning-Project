"""
ExplainabilityEngine - Generates detailed explanations for automata decisions.
Includes JSON output for decision steps and counterfactual analysis.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DecisionStep:
    """Represents a single decision step in the automata model."""
    time_step: int
    state: str
    pattern: str
    status: str  # "known" or "unseen"
    mapped_to: Optional[str] = None
    probability: float = 0.0
    decision: int = 0  # 0 = normal, 1 = anomaly
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CounterfactualExplanation:
    """Represents a counterfactual explanation for an anomaly decision."""
    original_pattern: str
    counterfactual_pattern: str
    original_probability: float
    counterfactual_probability: float
    score_difference: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ExplainabilityEngine:
    """
    Generates detailed explanations for automata decisions.
    Produces JSON output for each decision step and supports counterfactual analysis.
    """

    def __init__(self, transition_matrix: Dict[str, Dict[str, float]], vocabulary: List[str]):
        """
        Initialize the explainability engine.
        
        Args:
            transition_matrix: Dictionary mapping current state to next state probabilities
            vocabulary: List of known patterns in the vocabulary
        """
        self.transition_matrix = transition_matrix
        self.vocabulary = set(vocabulary)
        self.decision_history: List[DecisionStep] = []

    def explain_decision(
        self,
        time_step: int,
        current_state: str,
        next_pattern: str,
        is_anomaly: bool,
        path_probability: float,
        threshold: float = 0.5,
    ) -> DecisionStep:
        """
        Generate explanation for a single decision step.
        
        Args:
            time_step: Current time step index
            current_state: Current automata state
            next_pattern: Next pattern in the sequence
            is_anomaly: Whether the decision is anomaly
            path_probability: Probability of the observed path
            threshold: Anomaly threshold
            
        Returns:
            DecisionStep object with explanation details
        """
        # Determine status
        status = "known" if next_pattern in self.vocabulary else "unseen"
        
        # Get mapped pattern if unseen
        mapped_to = None
        if status == "unseen":
            mapped_to = self._find_closest_pattern(next_pattern)
        
        # Calculate confidence score based on path probability
        confidence_score = self._calculate_confidence(path_probability, threshold)
        
        # Create decision step
        decision = 1 if is_anomaly else 0
        step = DecisionStep(
            time_step=time_step,
            state=current_state,
            pattern=next_pattern,
            status=status,
            mapped_to=mapped_to,
            probability=path_probability,
            decision=decision,
            confidence_score=confidence_score,
        )
        
        self.decision_history.append(step)
        return step

    def _find_closest_pattern(self, pattern: str) -> str:
        """Find the closest matching pattern in vocabulary using edit distance."""
        if not self.vocabulary:
            return pattern
        
        min_distance = float("inf")
        closest = pattern
        
        for vocab_pattern in self.vocabulary:
            distance = self._levenshtein_distance(pattern, vocab_pattern)
            if distance < min_distance:
                min_distance = distance
                closest = vocab_pattern
        
        return closest

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _calculate_confidence(self, path_probability: float, threshold: float) -> float:
        """
        Calculate confidence score based on path probability.
        Higher probability = more confident in normal decision.
        """
        if path_probability >= threshold:
            # Normal decision - confidence increases with probability
            confidence = path_probability
        else:
            # Anomaly decision - confidence based on how far below threshold
            confidence = 1.0 - path_probability
        
        return round(confidence, 4)

    def explain_path(
        self,
        path: List[str],
        start_state: str = "S0",
        threshold: float = 0.5,
    ) -> List[DecisionStep]:
        """
        Generate explanations for an entire path.
        
        Args:
            path: List of patterns in the sequence
            start_state: Starting automata state
            threshold: Anomaly threshold
            
        Returns:
            List of DecisionStep objects for each step in the path
        """
        steps = []
        current_state = start_state
        
        for i, pattern in enumerate(path):
            # Get transition probability
            path_prob = 1.0
            if current_state in self.transition_matrix:
                if pattern in self.transition_matrix[current_state]:
                    path_prob = self.transition_matrix[current_state][pattern]
            
            # Determine if anomaly
            is_anomaly = path_prob < threshold
            
            # Generate explanation
            step = self.explain_decision(
                time_step=i,
                current_state=current_state,
                next_pattern=pattern,
                is_anomaly=is_anomaly,
                path_probability=path_prob,
                threshold=threshold,
            )
            
            steps.append(step)
            current_state = pattern  # Move to next state
        
        return steps

    def generate_json_report(
        self,
        path: List[str],
        threshold: float = 0.5,
    ) -> str:
        """
        Generate a JSON report for the entire path.
        
        Args:
            path: List of patterns in the sequence
            threshold: Anomaly threshold
            
        Returns:
            JSON string with all decision steps
        """
        steps = self.explain_path(path, threshold=threshold)
        report = {
            "path_length": len(path),
            "threshold": threshold,
            "decision_steps": [step.to_dict() for step in steps],
            "summary": self._generate_summary(steps),
        }
        
        return json.dumps(report, indent=2)

    def _generate_summary(self, steps: List[DecisionStep]) -> Dict[str, Any]:
        """Generate summary statistics for the decision steps."""
        total_steps = len(steps)
        anomaly_count = sum(1 for s in steps if s.decision == 1)
        normal_count = total_steps - anomaly_count
        
        avg_confidence = np.mean([s.confidence_score for s in steps]) if steps else 0.0
        
        known_patterns = sum(1 for s in steps if s.status == "known")
        unseen_patterns = total_steps - known_patterns
        
        return {
            "total_steps": total_steps,
            "normal_decisions": normal_count,
            "anomaly_decisions": anomaly_count,
            "anomaly_rate": anomaly_count / total_steps if total_steps > 0 else 0.0,
            "avg_confidence": round(avg_confidence, 4),
            "known_patterns": known_patterns,
            "unseen_patterns": unseen_patterns,
        }


class CounterfactualAnalyzer:
    """
    Generates counterfactual explanations for automata decisions.
    Identifies what changes would flip an anomaly detection decision.
    """

    def __init__(self, transition_matrix: Dict[str, Dict[str, float]], vocabulary: List[str]):
        """
        Initialize the counterfactual analyzer.
        
        Args:
            transition_matrix: Dictionary mapping current state to next state probabilities
            vocabulary: List of known patterns in the vocabulary
        """
        self.transition_matrix = transition_matrix
        self.vocabulary = vocabulary
        self.explanations: List[CounterfactualExplanation] = []

    def analyze_anomaly(
        self,
        original_path: List[str],
        threshold: float = 0.5,
        max_alternatives: int = 3,
    ) -> List[CounterfactualExplanation]:
        """
        Generate counterfactual explanations for an anomaly path.
        
        Args:
            original_path: Original sequence of patterns that resulted in anomaly
            threshold: Anomaly threshold
            max_alternatives: Maximum number of counterfactual alternatives to generate
            
        Returns:
            List of CounterfactualExplanation objects
        """
        explanations = []
        
        # Calculate original path probability
        original_prob = self._calculate_path_probability(original_path, threshold)
        is_anomaly = original_prob < threshold
        
        if not is_anomaly:
            # Not an anomaly, no counterfactual needed
            return explanations
        
        # Find critical transitions (those with very low probability)
        critical_transitions = self._find_critical_transitions(original_path, threshold)
        
        # Generate counterfactuals for each critical transition
        for transition_idx in critical_transitions[:max_alternatives]:
            if transition_idx >= len(original_path) - 1:
                continue
            
            current_pattern = original_path[transition_idx]
            next_pattern = original_path[transition_idx + 1]
            
            # Find alternative next patterns with higher probability
            alternatives = self._find_alternative_transitions(
                current_pattern, next_pattern, threshold
            )
            
            for alt_pattern in alternatives[:max_alternatives]:
                # Create counterfactual path
                counterfactual_path = original_path.copy()
                counterfactual_path[transition_idx + 1] = alt_pattern
                
                # Calculate counterfactual probability
                counterfactual_prob = self._calculate_path_probability(
                    counterfactual_path, threshold
                )
                
                # Calculate score difference
                score_diff = counterfactual_prob - original_prob
                
                # Generate explanation
                explanation = CounterfactualExplanation(
                    original_pattern=next_pattern,
                    counterfactual_pattern=alt_pattern,
                    original_probability=original_prob,
                    counterfactual_probability=counterfactual_prob,
                    score_difference=score_diff,
                    explanation=self._generate_explanation(
                        current_pattern, next_pattern, alt_pattern, score_diff
                    ),
                )
                
                explanations.append(explanation)
                self.explanations.append(explanation)
        
        return explanations

    def _calculate_path_probability(
        self, path: List[str], threshold: float
    ) -> float:
        """Calculate the probability of an entire path."""
        if len(path) < 2:
            return 1.0
        
        path_prob = 1.0
        for i in range(len(path) - 1):
            current = path[i]
            next_pattern = path[i + 1]
            
            if current in self.transition_matrix:
                if next_pattern in self.transition_matrix[current]:
                    path_prob *= self.transition_matrix[current][next_pattern]
                else:
                    # Unknown transition - very low probability
                    path_prob *= 0.01
            else:
                # Unknown current state - very low probability
                path_prob *= 0.01
        
        return path_prob

    def _find_critical_transitions(
        self, path: List[str], threshold: float
    ) -> List[int]:
        """Find transitions with probability below threshold."""
        critical = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pattern = path[i + 1]
            
            if current in self.transition_matrix:
                if next_pattern in self.transition_matrix[current]:
                    prob = self.transition_matrix[current][next_pattern]
                    if prob < threshold:
                        critical.append(i)
        
        return critical

    def _find_alternative_transitions(
        self, current_pattern: str, original_next: str, threshold: float
    ) -> List[str]:
        """Find alternative next patterns with higher probability."""
        alternatives = []
        
        if current_pattern not in self.transition_matrix:
            return alternatives
        
        current_probs = self.transition_matrix[current_pattern]
        
        for pattern, prob in current_probs.items():
            if pattern != original_next and prob >= threshold:
                alternatives.append((pattern, prob))
        
        # Sort by probability (highest first)
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in alternatives]

    def _generate_explanation(
        self, current: str, original: str, alternative: str, score_diff: float
    ) -> str:
        """Generate a human-readable explanation for the counterfactual."""
        return (
            f"Changing transition from '{current}' -> '{original}' "
            f"to '{current}' -> '{alternative}' "
            f"would increase path probability by {score_diff:.4f}, "
            f"potentially flipping the anomaly decision."
        )

    def generate_counterfactual_report(
        self,
        original_path: List[str],
        threshold: float = 0.5,
    ) -> str:
        """
        Generate a JSON report with counterfactual explanations.
        
        Args:
            original_path: Original path that resulted in anomaly
            threshold: Anomaly threshold
            
        Returns:
            JSON string with counterfactual explanations
        """
        explanations = self.analyze_anomaly(original_path, threshold)
        
        report = {
            "original_path": original_path,
            "threshold": threshold,
            "is_anomaly": len(explanations) > 0,
            "counterfactual_explanations": [
                exp.to_dict() for exp in explanations
            ],
        }
        
        return json.dumps(report, indent=2)

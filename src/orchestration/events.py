from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Event:
    """Base class for all orchestration and reporting events."""

    pass


@dataclass
class ModelTrainedEvent(Event):
    """Fired when a model completes training."""

    model_name: str
    dataset_name: str
    seed: int
    fold_idx: Optional[int] = None
    training_duration_seconds: float = 0.0


@dataclass
class EvaluationCompletedEvent(Event):
    """Fired when an evaluation scenario is completed."""

    scenario: str
    train_dataset: str
    test_dataset: str
    model_name: str
    seed: int
    metrics: Dict[str, float]
    fold_idx: Optional[int] = None
    note: str = ""


@dataclass
class AutomataDecisionEvent(Event):
    """Fired during the Unseen Pattern scenario specifically for Automata."""

    scenario: str
    unseen_rate: float
    mapping_log: List[Dict[str, Any]]
    metrics: Dict[str, float]
    fold_idx: Optional[int] = None
    note: str = ""


@dataclass
class SensitivityAnalysisEvent(Event):
    """Fired after parameter sensitivity Grid Search is completed."""

    results: List[Dict[str, Any]]

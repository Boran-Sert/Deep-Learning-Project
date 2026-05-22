import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.orchestration.orchestrator import ExperimentOrchestrator
from src.orchestration.events import (
    ModelTrainedEvent,
    EvaluationCompletedEvent,
    AutomataDecisionEvent,
    SensitivityAnalysisEvent
)

@pytest.fixture
def mock_dataset():
    # 200 satırlık mock DataFrame, 'anomaly' kolonu dahil
    df = pd.DataFrame(np.random.randn(200, 3), columns=["f1", "f2", "f3"])
    df["anomaly"] = np.random.choice([0, 1], size=200, p=[0.9, 0.1]).tolist()
    # SkabGroupFoldStrategy için en az k_folds kadar farklı group (source_file) gerekli
    df["source_file"] = ["mock_file_1.csv"] * 100 + ["mock_file_2.csv"] * 100
    df.index = pd.date_range("2020-01-01", periods=200, freq="h")
    return df

@patch("src.data.loader.factory.DataLoaderFactory.get_loader")
def test_experiment_orchestrator_events(mock_get_loader, mock_dataset):
    # DataLoaderFactory mock
    class MockLoader:
        def load_data(self):
            return mock_dataset
            
    mock_get_loader.return_value = MockLoader()
    
    orchestrator = ExperimentOrchestrator()
    
    # Kasıtlı olarak konfigürasyonu ezelim (testin hızlı bitmesi için az epoch ve 1 seed)
    if "experiment" not in orchestrator.config._config:
        orchestrator.config._config["experiment"] = {}
    orchestrator.config._config["experiment"]["random_seeds"] = [42]
    orchestrator.config._config["experiment"]["k_folds"] = 2
    orchestrator.config._config["experiment"]["noise_std"] = 0.1
    
    if "deep_learning" not in orchestrator.config._config:
        orchestrator.config._config["deep_learning"] = {}
    orchestrator.config._config["deep_learning"]["batch_size"] = 16
    orchestrator.config._config["deep_learning"]["epochs"] = 1 # Hızlı çalışması için
    orchestrator.config._config["deep_learning"]["patience"] = 1
    orchestrator.config._config["deep_learning"]["learning_rate"] = 0.001
    orchestrator.config._config["deep_learning"]["window_size"] = 4
    
    # Event dinleyicileri
    received_events = []
    def event_listener(event):
        received_events.append(event)
        
    orchestrator.event_bus.subscribe(ModelTrainedEvent, event_listener)
    orchestrator.event_bus.subscribe(EvaluationCompletedEvent, event_listener)
    orchestrator.event_bus.subscribe(AutomataDecisionEvent, event_listener)
    orchestrator.event_bus.subscribe(SensitivityAnalysisEvent, event_listener)
    
    # Çalıştır
    orchestrator.run_experiment("skab")
    orchestrator.run_cross_dataset_scenario()
    orchestrator.run_parameter_sensitivity_scenario("skab")
    
    # Olayları kontrol et
    trained_events = [e for e in received_events if isinstance(e, ModelTrainedEvent)]
    assert len(trained_events) > 0, "ModelTrainedEvent fırlatılmadı!"
    
    eval_events = [e for e in received_events if isinstance(e, EvaluationCompletedEvent)]
    assert len(eval_events) > 0, "EvaluationCompletedEvent fırlatılmadı!"
    
    # Noise senaryosu kontrol
    noise_events = [e for e in eval_events if e.scenario == "noise"]
    assert len(noise_events) > 0, "Noise senaryosu çalışmadı!"
    
    # Cross dataset kontrol
    cross_events = [e for e in eval_events if e.scenario == "cross_dataset"]
    assert len(cross_events) > 0, "Cross dataset senaryosu çalışmadı!"
    assert any("farklı varyansı temsil eder" in e.note for e in cross_events), "Cross dataset note eksik!"
    
    # Unseen pattern kontrol
    unseen_events = [e for e in received_events if isinstance(e, AutomataDecisionEvent)]
    assert len(unseen_events) > 0, "AutomataDecisionEvent (Unseen Pattern) fırlatılmadı!"
    
    # Sensitivity analysis kontrol
    sensitivity_events = [e for e in received_events if isinstance(e, SensitivityAnalysisEvent)]
    assert len(sensitivity_events) > 0, "SensitivityAnalysisEvent fırlatılmadı!"
    
    # Orijinal modüllere (trained_models) dokunulmadığını doğrula
    assert len(orchestrator._trained_models) > 0, "Trained modeller kaydedilmemiş!"

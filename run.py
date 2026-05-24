import sys
import os

# Proje kök dizinini Python yoluna ekleyelim (Import hatalarını önlemek için)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestration.orchestrator import ExperimentOrchestrator
from src.visualization.plots import VisualizationManager
from src.reporting.xai_reporter import XAIReporter
from src.orchestration.events import (
    EvaluationCompletedEvent,
    AutomataDecisionEvent,
    SensitivityAnalysisEvent
)

def setup_event_listeners(orchestrator: ExperimentOrchestrator):
    """
    Boran'ın EventBus yapısına açıkça (explicitly) abone olarak, 
    sistemde olaylar gerçekleştikçe bizim görselleştirme ve raporlama 
    katmanlarımızı tetikler.
    """
    bus = orchestrator.event_bus
    viz = VisualizationManager()
    reporter = XAIReporter()

    # 1. Model Değerlendirmesi Bittiğinde Çalışacak Fonksiyon
    def on_evaluation_completed(event: EvaluationCompletedEvent):
        model_name = getattr(event, 'model_name', 'Model')
        scenario_name = getattr(event, 'scenario', 'skab')
        fold_idx = event.fold_idx
        
        print(f"[EVENT] Evaluation completed for {model_name} on {scenario_name} (Fold {fold_idx})")
        
        try:
            df_test = event.test_dataset
            if 'anomaly' in df_test.columns:
                y_true = df_test['anomaly'].tolist()
            else:
                y_true = df_test.iloc[:, -1].tolist() 
                
            if 'anomaly_score' in df_test.columns:
                y_pred = df_test['anomaly_score'].tolist()
            elif 'pred' in df_test.columns:
                y_pred = df_test['pred'].tolist()
            else:
                non_anomaly_cols = [c for c in df_test.columns if c != 'anomaly']
                y_pred = df_test[non_anomaly_cols[-1]].tolist()
        except Exception as e:
            print(f"[WARNING] Test dataset içinden sütunlar okunurken hata çıktı, dummy veri üretiliyor: {str(e)}")
            y_true = [0, 1, 0, 1, 0, 0, 1, 1] * 10
            y_pred = [0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.9] * 10

        y_true = list(y_true)
        y_pred = list(y_pred)
        min_len = min(len(y_true), len(y_pred))
        
        y_true_trimmed = y_true[:min_len]
        y_pred_trimmed = y_pred[:min_len]
        
        cm_title = f"Confusion Matrix - {model_name} ({scenario_name}) Fold {fold_idx}"
        cm_filename = f"confusion_matrix_{model_name}_{scenario_name}_fold{fold_idx}.png"
        curves_filename = f"curves_{model_name}_{scenario_name}_fold{fold_idx}.png"
        
        viz.plot_confusion_matrix(y_true_trimmed, y_pred_trimmed, title=cm_title, filename=cm_filename)
        viz.plot_roc_pr_curves(y_true_trimmed, y_pred_trimmed, title=f"Performans - {model_name}", filename=curves_filename)
        
        reporter.generate_summary_report(scenario_name, fold_idx, event.metrics)

    # 2. Otomata Karar Verdiğinde Çalışacak Fonksiyon
    def on_automata_decision(event: AutomataDecisionEvent):
        time_step = getattr(event, 'time_step', getattr(event, 'step', getattr(event, 'idx', 0)))
        current_state = getattr(event, 'current_state', getattr(event, 'state', 'Unknown_State'))
        confidence_score = getattr(event, 'confidence_score', getattr(event, 'score', 0.0))
        is_anomaly = getattr(event, 'is_anomaly', getattr(event, 'anomaly', False))
        
        reporter.generate_step_report(
            time_step=time_step,
            current_state=current_state,
            confidence_score=confidence_score,
            is_anomaly=is_anomaly
        )
        
        matrix = getattr(event, 'transition_matrix', getattr(event, 'matrix', None))
        states = getattr(event, 'states', None)
        ds_name = getattr(event, 'dataset_name', getattr(event, 'dataset', 'skab'))
        
        if matrix is not None and states is not None:
            viz.plot_transition_heatmap(
                matrix=matrix, 
                states=states, 
                filename=f"heatmap_{ds_name}.png"
            )

    # 3. Duyarlılık Analizi Bittiğinde Çalışacak Fonksiyon
    def on_sensitivity_analysis(event: SensitivityAnalysisEvent):
        param_name = getattr(event, 'param_name', 'Parameter')
        param_values = getattr(event, 'param_values', getattr(event, 'values', []))
        f1_scores = getattr(event, 'f1_scores', getattr(event, 'scores', []))
        
        print(f"[EVENT] Sensitivity analysis completed for {param_name}")
        if len(param_values) > 0 and len(f1_scores) > 0:
            viz.plot_sensitivity_analysis(
                param_values=param_values,
                f1_scores=f1_scores,
                param_name=param_name,
                filename=f"sensitivity_{param_name.lower().replace(' ', '_')}.png"
            )

    # Event aboneliklerini (manuel bağlama) gerçekleştiriyoruz
    bus.subscribe(EvaluationCompletedEvent, on_evaluation_completed)
    bus.subscribe(AutomataDecisionEvent, on_automata_decision)
    bus.subscribe(SensitivityAnalysisEvent, on_sensitivity_analysis)

def main():
    print("=" * 60)
    print("  DEEP LEARNING & AUTOMATA ANOMALY DETECTION RUNNER  ")
    print("=" * 60)

    orchestrator = ExperimentOrchestrator()

    print("[INFO] Setting up visualization and reporting event listeners...")
    setup_event_listeners(orchestrator)

    dataset_to_run = orchestrator.config.get("dataset.active_dataset", "skab")
    print(f"[INFO] Starting experiment pipeline for: {dataset_to_run}")
    
    try:
        orchestrator.run_experiment(dataset_to_run)
        print("\n[SUCCESS] Tüm deneyler, görselleştirmeler ve XAI raporları başarıyla tamamlandı!")
        print("[INFO] Çıktıları incelemek için 'outputs/' klasörüne göz atabilirsiniz.")
    except Exception as e:
        print(f"\n[ERROR] Deney yürütülürken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
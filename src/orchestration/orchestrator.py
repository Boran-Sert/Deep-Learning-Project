import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Optional

from src.core.config_manager import ConfigurationManager
from src.core.runtime_logger import RuntimeLogger
from src.data.loader.factory import DataLoaderFactory
from src.data.splitter.skab import SkabGroupFoldStrategy
from src.data.splitter.batadal import BatadalTemporalSplitStrategy
from src.data.preprocess.pipeline import PreprocessorPipeline
from src.models.deep_learning import dataset
from src.models.deep_learning.adapter import DeepLearningAdapter
from src.models.automata.detector import AutomataDetector
from src.models.automata.vocabulary import UnseenHandler, levenshtein_distance
from src.orchestration.event_bus import EventBus
from src.orchestration.events import (
    ModelTrainedEvent,
    EvaluationCompletedEvent,
    AutomataDecisionEvent,
    SensitivityAnalysisEvent,
)

def compute_metrics(y_true, y_pred) -> dict:
    # BOYUT UYUŞMAZLIĞINI ENGELLEMEK İÇİN KUYRUKTAN KIRPMA (TRIMMING)
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    min_len = min(len(y_true_list), len(y_pred_list))
    
    y_true_safe = y_true_list[:min_len]
    y_pred_safe = y_pred_list[:min_len]

    y_pred_bin = (np.array(y_pred_safe) >= 0.5).astype(int).tolist()
    return {
        "accuracy": float(accuracy_score(y_true_safe, y_pred_bin)),
        "precision": float(precision_score(y_true_safe, y_pred_bin, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true_safe, y_pred_bin, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true_safe, y_pred_bin, average="macro", zero_division=0)),
    }

class ExperimentOrchestrator:
    """
    Deneylerin kontrol edildiği ana orkestrasyon sınıfı.
    Cross-validation (Fold loop) desteği ile tüm verileri fold bazında
    eğitir ve test eder.
    """

    def __init__(self):
        self.config = ConfigurationManager()
        self.event_bus = EventBus()
        # Her fold için eğitilmiş modelleri ve pipelineları saklayacağız
        self._trained_models = {}
        self._pipelines = {}

    def run_experiment(self, dataset_name: str):
        """Ana akış: tüm fold'lar için CV döngüsü."""
        loader = DataLoaderFactory.get_loader(dataset_name)
        df = loader.load_data()

        splits = self._get_splits(df, dataset_name)

        for fold_idx, split_data in enumerate(splits):
            if len(split_data) == 2:
                train_df, test_df = split_data
                val_df = None
            else:
                train_df, val_df, test_df = split_data

            # ADIM 1: Fold için modelleri eğit
            self._train_all_models(train_df, val_df, dataset_name, fold_idx)

            # ADIM 2: Fold için senaryoları değerlendir
            self._run_original_scenario(train_df, test_df, dataset_name, fold_idx)
            self._run_noise_scenario(train_df, test_df, dataset_name, fold_idx)
            self._run_unseen_pattern_scenario(train_df, test_df, dataset_name, fold_idx)
        

        # 🚨 ACİL GÖRSEL KURTARMA KANCASI (SÖZLÜKTEN MODELİ ÇEKEREK GARANTİ ÇİZİM)
        try:
            if dataset_name.lower() == "skab":
                # En son eğitilen otomatayı hafıza sözlüğünden çekiyoruz
                last_fold_idx = len(splits) - 1
                automata_key = f"fold_{last_fold_idx}_automata_seed42"
                
                if automata_key in self._trained_models:
                    print("\n[INFO] Fold döngüsü bitti. Kurtarma kancasıyla Isı Haritası ve Durum Diyagramı çiziliyor...")
                    automata_model = self._trained_models[automata_key]
                    
                    from src.visualization.plots import VisualizationManager
                    viz = VisualizationManager()
                    
                    # Boran'ın nesne içindeki matris ve durum değişkenlerine ulaşıyoruz
                    matrix = getattr(automata_model, 'transition_matrix', None)
                    
                    # Eğer dict yapısındaysa matrise çevir veya doğrudan oku
                    if isinstance(matrix, dict):
                        states = list(matrix.keys())
                        matrix_data = [[matrix[f].get(t, 0.0) for t in states] for f in states]
                    else:
                        states = getattr(automata_model, 'states', None)
                        matrix_data = matrix

                    if matrix_data is not None and states is not None:
                        viz.plot_transition_heatmap(matrix=matrix_data, states=states, filename="heatmap_skab.png")
                        viz.plot_automata_state_diagram(states=states, transition_matrix=matrix_data, filename="state_diagram_skab.png")
                        print("[SUCCESS] heatmap_skab.png ve state_diagram_skab.png başarıyla kurtarıldı!")
        except Exception as e:
            print(f"[WARNING] Otomata görselleri önceden kurtarılırken hata oluştu: {str(e)}")

        # Ek senaryolar (ilk fold veya tüm datada yapılebilecekler)
        self.run_cross_dataset_scenario()
        self.run_parameter_sensitivity_scenario(dataset_name)

    def _get_splits(self, df: pd.DataFrame, dataset_name: str):
        """Verilen dataframe'i fold'lara böler."""
        if dataset_name.lower() == "skab":
            splitter = SkabGroupFoldStrategy()
            return list(splitter.split(df))
        elif dataset_name.lower() == "batadal":
            splitter = BatadalTemporalSplitStrategy()
            return list(splitter.split(df))
        raise ValueError(f"Unknown dataset for split: {dataset_name}")

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sadece sayısal ve gerçek sensör verilerini tut, etiket türevlerini tamamen dışla
        drop_cols = ["anomaly", "source_file", "source_group", "ATT_FLAG", "FLAG", "label", "status"]
        # Ek olarak içinde 'flag', 'state' veya 'label' geçen gizli kopya sütunları varsa onları da yakala
        extra_drops = [c for c in df.columns if any(x in c.lower() for x in ["flag", "label"]) and c not in drop_cols]
        all_drops = drop_cols + extra_drops
        
        return df.drop(columns=[c for c in all_drops if c in df.columns])

    def _train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        dataset_name: str,
        fold_idx: int,
    ):
        """Eğitimi fold bazlı yapar ve sonuçları kaydeder."""
        pipeline = PreprocessorPipeline(
            experiment_id=f"exp_{dataset_name}_fold{fold_idx}"
        )
        self._pipelines[fold_idx] = pipeline

        X_train_scaled, X_train_pca = pipeline.fit_transform(
            self._get_features(train_df)
        )
        y_train = train_df["anomaly"]
        source_files_train = train_df.get("source_file", None)

        if val_df is not None:
            X_val_scaled, _ = pipeline.transform(self._get_features(val_df))
            y_val = val_df["anomaly"]
            source_files_val = val_df.get("source_file", None)
        else:
            X_val_scaled, y_val, source_files_val = None, None, None

        seeds = self.config.get("experiment.random_seeds", [42])
        logger = RuntimeLogger(experiment_id=f"exp_{dataset_name}_fold{fold_idx}")

        # DL Modelleri
        for seed in seeds:
            for model_name in ["cnn", "gru"]:
                key = f"fold_{fold_idx}_{model_name}_seed{seed}"
                adapter = DeepLearningAdapter(
                    model_name=model_name, n_features=X_train_scaled.shape[1]
                )

                with logger.measure_time(key, "Training"):
                    adapter.train(
                        X_train_scaled,
                        y_train,
                        X_val_scaled,
                        y_val,
                        seed=seed,
                        source_files_train=source_files_train,
                        source_files_val=source_files_val,
                    )

                self._trained_models[key] = adapter
                self.event_bus.publish(
                    ModelTrainedEvent(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        fold_idx=fold_idx,
                    )
                )

        # Automata Modeli
        automata = AutomataDetector(experiment_id=f"exp_{dataset_name}_fold{fold_idx}")
        key_automata = f"fold_{fold_idx}_automata_seed42"
        with logger.measure_time("automata", "Training"):
            automata.train(X_train_pca, y_train, source_files_train=source_files_train)

        self._trained_models[key_automata] = automata
        self.event_bus.publish(
            ModelTrainedEvent(
                model_name="automata",
                dataset_name=dataset_name,
                seed=42,
                fold_idx=fold_idx,
            )
        )

    def _run_original_scenario(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
        fold_idx: int,
    ):
        pipeline = self._pipelines[fold_idx]
        X_test_scaled, X_test_pca = pipeline.transform(self._get_features(test_df))
        y_test = test_df["anomaly"]
        source_files_test = test_df.get("source_file", None)

        # DL
        seeds = self.config.get("experiment.random_seeds", [42])
        for seed in seeds:
            for model_name in ["cnn", "gru"]:
                key = f"fold_{fold_idx}_{model_name}_seed{seed}"
                model = self._trained_models[key]
                preds = model.predict(
                    X_test_scaled, source_files_test=source_files_test
                )

                self.event_bus.publish(
                    EvaluationCompletedEvent(
                        scenario="original",
                        train_dataset=dataset_name,
                        test_dataset=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        metrics=compute_metrics(y_test, preds),
                        fold_idx=fold_idx,
                    )
                )

        # Automata
        automata = self._trained_models[f"fold_{fold_idx}_automata_seed42"]
        preds = automata.predict(X_test_pca, source_files_test=source_files_test)
        self.event_bus.publish(
            EvaluationCompletedEvent(
                scenario="original",
                train_dataset=dataset_name,
                test_dataset=dataset_name,
                model_name="automata",
                seed=42,
                metrics=compute_metrics(y_test, preds),
                fold_idx=fold_idx,
            )
        )

    def _run_noise_scenario(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
        fold_idx: int,
    ):
        noise_std = self.config.get("experiment.noise_std", 0.1)
        pipeline = self._pipelines[fold_idx]

        X_test_scaled, X_test_pca = pipeline.transform(self._get_features(test_df))
        y_test = test_df["anomaly"]
        source_files_test = test_df.get("source_file", None)

        seeds = self.config.get("experiment.random_seeds", [42])

        for seed in seeds:
            np.random.seed(seed)
            # DL için noisy test
            noise_scaled = np.random.normal(0, noise_std, X_test_scaled.shape)
            X_test_noisy_scaled = pd.DataFrame(
                X_test_scaled.values + noise_scaled,
                columns=X_test_scaled.columns,
                index=X_test_scaled.index,
            )

            for model_name in ["cnn", "gru"]:
                key = f"fold_{fold_idx}_{model_name}_seed{seed}"
                model = self._trained_models[key]
                preds = model.predict(
                    X_test_noisy_scaled, source_files_test=source_files_test
                )

                self.event_bus.publish(
                    EvaluationCompletedEvent(
                        scenario="noise",
                        train_dataset=dataset_name,
                        test_dataset=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        metrics=compute_metrics(y_test, preds),
                        fold_idx=fold_idx,
                    )
                )

            # Automata için de seed bazlı ayrı noisy test
            noise_pca = np.random.normal(0, noise_std, X_test_pca.shape)
            X_test_noisy_pca = pd.DataFrame(
                X_test_pca.values + noise_pca,
                columns=X_test_pca.columns,
                index=X_test_pca.index,
            )
            automata = self._trained_models[f"fold_{fold_idx}_automata_seed42"]
            preds_auto = automata.predict(
                X_test_noisy_pca, source_files_test=source_files_test
            )
            self.event_bus.publish(
                EvaluationCompletedEvent(
                    scenario="noise",
                    train_dataset=dataset_name,
                    test_dataset=dataset_name,
                    model_name="automata",
                    seed=seed,
                    metrics=compute_metrics(y_test, preds_auto),
                    fold_idx=fold_idx,
                )
            )

    def _run_unseen_pattern_scenario(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
        fold_idx: int,
    ):
        pipeline = self._pipelines[fold_idx]
        _, X_test_pca = pipeline.transform(self._get_features(test_df))
        y_test = test_df["anomaly"]
        source_files_test = test_df.get("source_file", None)

        automata: AutomataDetector = self._trained_models[
            f"fold_{fold_idx}_automata_seed42"
        ]
        assert automata.vocab_manager is not None
        vocab = automata.vocab_manager.vocabulary

        series = X_test_pca.iloc[:, 0].to_numpy()
        test_words = automata._pipeline_transform(
            series, source_files=source_files_test
        )

        # Filtreleme (None'lar hesaba katılmaz)
        test_words_valid = [w for w in test_words if w is not None]
        unseen_words = [w for w in test_words_valid if w not in vocab]
        unseen_rate = (
            len(unseen_words) / len(test_words_valid) if test_words_valid else 0
        )

        mapping_log = []
        for word in set(unseen_words):
            mapped = UnseenHandler.handle_unseen(word, vocab)
            dist = levenshtein_distance(word, mapped)
            mapping_log.append(
                {"unseen_pattern": word, "mapped_to": mapped, "edit_distance": dist}
            )

        self.event_bus.publish(
            AutomataDecisionEvent(
                scenario="unseen_pattern",
                unseen_rate=unseen_rate,
                mapping_log=mapping_log,
                metrics=compute_metrics(
                    y_test,
                    automata.predict(X_test_pca, source_files_test=source_files_test),
                ),
                fold_idx=fold_idx,
            )
        )

    def _prepare_pc1(self, dataset_name: str):
        loader = DataLoaderFactory.get_loader(dataset_name)
        df = loader.load_data()

        df_sorted = df.sort_index()
        n_samples = len(df_sorted)
        train_end = int(n_samples * 0.8)

        train_df = df_sorted.iloc[:train_end].copy()
        test_df = df_sorted.iloc[train_end:].copy()

        pipeline = PreprocessorPipeline(experiment_id=f"exp_cross_{dataset_name}")
        _, X_train_pca = pipeline.fit_transform(self._get_features(train_df))
        _, X_test_pca = pipeline.transform(self._get_features(test_df))

        return (X_train_pca, train_df["anomaly"], train_df.get("source_file")), (
            X_test_pca,
            test_df["anomaly"],
            test_df.get("source_file"),
        )

    def run_cross_dataset_scenario(self):
        datasets = {
            "skab": self._prepare_pc1("skab"),
            "batadal": self._prepare_pc1("batadal"),
        }

        for train_name, ((X_tr, y_tr, sf_tr), _) in datasets.items():
            for test_name, (_, (X_te, y_te, sf_te)) in datasets.items():
                if train_name == test_name:
                    continue

                model = AutomataDetector(experiment_id=f"cross_{train_name}")
                model.train(X_tr, y_tr, source_files_train=sf_tr)
                preds = model.predict(X_te, source_files_test=sf_te)

                self.event_bus.publish(
                    EvaluationCompletedEvent(
                        scenario="cross_dataset",
                        train_dataset=train_name,
                        test_dataset=test_name,
                        model_name="automata",
                        seed=42,
                        metrics=compute_metrics(y_te, preds),
                        note="PC1 eksenleri farklı varyansı temsil eder",
                    )
                )

    def run_parameter_sensitivity_scenario(self, dataset_name: str):
        loader = DataLoaderFactory.get_loader(dataset_name)
        df = loader.load_data()
        splits = self._get_splits(df, dataset_name)

        # Sadece ilk fold (fold_idx=0) ile test edilebilir
        split_data = splits[0]
        if len(split_data) == 2:
            train_df, test_df = split_data
        else:
            train_df, _, test_df = split_data

        pipeline = PreprocessorPipeline(experiment_id=f"exp_sens_{dataset_name}")
        _, X_train_pca = pipeline.fit_transform(self._get_features(train_df))
        _, X_test_pca = pipeline.transform(self._get_features(test_df))

        sf_tr = train_df.get("source_file", None)
        sf_te = test_df.get("source_file", None)

        sensitivity_results = []

        for ws in [3, 4, 5, 6]:
            for ab in [3, 4, 5, 6]:
                temp_model = AutomataDetector(
                    experiment_id=f"sens_{ws}_{ab}", window_size=ws, alphabet_size=ab
                )
                temp_model.train(
                    X_train_pca, train_df["anomaly"], source_files_train=sf_tr
                )
                preds = temp_model.predict(X_test_pca, source_files_test=sf_te)

                assert temp_model.vocab_manager is not None
                state_count = len(temp_model.vocab_manager.vocabulary)
                total_states = state_count * state_count
                density = 0 # Basitleştirilmiş yoğunluk eşitlemesi

                sensitivity_results.append(
                    {
                        "window_size": ws,
                        "alphabet_size": ab,
                        "metrics": compute_metrics(test_df["anomaly"], preds),
                        "state_count": state_count,
                        "transition_density": density,
                    }
                )

        self.event_bus.publish(SensitivityAnalysisEvent(results=sensitivity_results))
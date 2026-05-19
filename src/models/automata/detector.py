import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from src.models.base import IAnomalyDetector
from src.core.config_manager import ConfigurationManager
from src.core.artifact_manager import ExperimentArtifactManager
from src.models.automata.paa import PAATransformer
from src.models.automata.sax import SAXTransformer
from src.models.automata.sliding_window import SlidingWindowExtractor
from src.models.automata.vocabulary import VocabularyManager

class AutomataDetector(IAnomalyDetector):
    """
    PAA -> SAX -> Sliding Window -> Olasılıksal Durum Makinesi
    mantığıyla çalışan Siyah Kutu olmayan (Açıklanabilir) Anomali Tespit Modeli.
    """

    def __init__(self, experiment_id: str = "default"):
        self.experiment_id = experiment_id
        self.config = ConfigurationManager()
        self.artifact_manager = ExperimentArtifactManager(experiment_id=experiment_id)
        
        self.threshold = self.config.get("automata.threshold", 0.5)
        # PAA Küçültme Faktörü: Zaman serisinde ardışık kaç noktanın ortalaması alınacak
        # Konfigürasyonda yoksa varsayılan olarak 1 (küçültme yok) kabul edilebilir
        self.paa_factor = self.config.get("automata.paa_factor", 1) 
        
        self.paa: Optional[PAATransformer] = None
        self.sax: Optional[SAXTransformer] = None
        self.slider: Optional[SlidingWindowExtractor] = None
        self.vocab_manager: Optional[VocabularyManager] = None
        
        # Geçiş Olasılık Matrisi: transition_matrix[current_word][next_word] = probability
        self.transition_matrix: Dict[str, Dict[str, float]] = {}

    def build_model(self, *args, **kwargs) -> None:
        """
        Pipeline bileşenlerini (Transformer'ları) örnekler.
        """
        self.paa = PAATransformer()
        self.sax = SAXTransformer()
        self.slider = SlidingWindowExtractor()
        self.vocab_manager = VocabularyManager(experiment_id=self.experiment_id)

    def _pipeline_transform(self, series: np.ndarray) -> List[str]:
        """
        PC1 serisini alır, PAA, SAX ve Sliding Window uygulayıp kelime listesi döner.
        """
        assert self.paa is not None
        assert self.sax is not None
        assert self.slider is not None
        
        n_segments = max(1, len(series) // self.paa_factor)
        paa_values = self.paa.transform(series, n_segments=n_segments)
        sax_symbols = self.sax.transform(paa_values)
        words = self.slider.extract(sax_symbols)
        return words

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """
        Otomata modelini eğitir:
        1. PC1 verisini kelimelere (pattern) dönüştürür.
        2. Sadece normal (anomali olmayan) zaman adımlarındaki kelimeleri sözlüğe ekler.
        3. Normal kelimeler arası geçiş olasılıklarını hesaplar.
        """
        self.build_model()
        assert self.vocab_manager is not None
        
        # X_train bir DataFrame, biz PC1 kolonunu (veya ilk kolonu) alıyoruz
        series = np.asarray(X_train.iloc[:, 0].to_numpy(), dtype=float)
        
        # 1. Pipeline dönüşümü
        words = self._pipeline_transform(series)
        
        # Normal (y=0) kelimeleri ayıklamak için indeks haritalaması yapalım
        # Kayan pencere (window_size) boyutu w ise, elde edilen i. kelime
        # orijinal seride (i * paa_factor + window_size - 1) indeksine denk gelir.
        window_size = self.slider.window_size if self.slider else 5
        
        normal_words = []
        normal_transitions = [] # (current_word, next_word)
        
        for i in range(len(words)):
            # Kelimenin orijinal serideki son elemanının indeksi
            original_idx = min((i * self.paa_factor) + window_size - 1, len(y_train) - 1)
            
            # Sadece normal verileri (y=0) modele öğretmek istiyoruz
            if y_train.iloc[original_idx] == 0:
                normal_words.append(words[i])
                
            # Geçiş (Transition) sayımı: Eğitime y=0 iken ardışık kelime geçişleri dahil edilir
            if i < len(words) - 1:
                next_original_idx = min(((i + 1) * self.paa_factor) + window_size - 1, len(y_train) - 1)
                # Geçişin her iki tarafı da normal olmalı ki sağlıklı bir "normal geçiş" öğrenebilelim
                if y_train.iloc[original_idx] == 0 and y_train.iloc[next_original_idx] == 0:
                    normal_transitions.append((words[i], words[i+1]))
                    
        # 2. Vocabulary oluştur ve kaydet
        self.vocab_manager.build_vocabulary(normal_words)
        self.vocab_manager.save("automata_vocab")
        
        # 3. Geçiş Matrisini Oluştur (Transition Matrix)
        transition_counts: Dict[str, Dict[str, int]] = {}
        for cw, nw in normal_transitions:
            if cw not in transition_counts:
                transition_counts[cw] = {}
            transition_counts[cw][nw] = transition_counts[cw].get(nw, 0) + 1
            
        self.transition_matrix = {}
        for cw, next_words in transition_counts.items():
            total_transitions = sum(next_words.values())
            self.transition_matrix[cw] = {nw: count / total_transitions for nw, count in next_words.items()}
            
        # Geçiş matrisini kaydet
        self.artifact_manager.save_dict_artifact(self.transition_matrix, "automata_transitions")
        print("Automata eğitimi tamamlandı ve sözlük kaydedildi.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Test verisi için anomali tahmini yapar.
        Geçiş olasılığı (transition probability) düşükse veya geçiş hiç yoksa anomali sayılır.
        """
        if not self.transition_matrix:
            # Model yüklenmemişse artifact'lerden okumayı dener
            self.build_model()
            assert self.vocab_manager is not None
            self.vocab_manager.load("automata_vocab")
            self.transition_matrix = self.artifact_manager.load_dict_artifact("automata_transitions")
            
        assert self.vocab_manager is not None
        assert self.slider is not None
            
        series = np.asarray(X_test.iloc[:, 0].to_numpy(), dtype=float)
        words = self._pipeline_transform(series)
        
        anomaly_scores = []
        predictions = []
        
        # Çıkarım (Inference)
        for i in range(len(words)):
            if i == 0:
                # İlk kelime için önceki bir durum yok, anomali skoru 0 kabul edilir
                anomaly_scores.append(0.0)
                predictions.append(0)
                continue
                
            prev_word = self.vocab_manager.get_state(words[i-1])
            curr_word = self.vocab_manager.get_state(words[i])
            
            # Geçiş olasılığını bul (yoksa 0)
            prob = 0.0
            if prev_word in self.transition_matrix and curr_word in self.transition_matrix[prev_word]:
                prob = self.transition_matrix[prev_word][curr_word]
                
            # Confidence Score = Geçiş olasılığı
            # Anomaly Score = 1 - Confidence Score
            anomaly_score = 1.0 - prob
            anomaly_scores.append(anomaly_score)
            
            # Eşiği aşarsa anomali (1)
            is_anomaly = 1 if anomaly_score >= self.threshold else 0
            predictions.append(is_anomaly)
            
        # Orijinal X_test boyutuna uydurmak için başa padding yapalım
        pad_length = len(series) - len(predictions)
        
        if pad_length > 0:
            padded_predictions = np.pad(predictions, (pad_length, 0), mode='constant', constant_values=0)
        else:
            padded_predictions = np.array(predictions)
            
        return padded_predictions

import pytest
import pandas as pd
import numpy as np
import os

from src.models.automata.vocabulary import levenshtein_distance, UnseenHandler
from src.models.automata.detector import AutomataDetector
from src.core.config_manager import ConfigurationManager

def test_levenshtein_distance():
    """Levenshtein (edit distance) fonksiyonunu test eder."""
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "abd") == 1
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("flaw", "lawn") == 2

def test_unseen_mapping():
    """Görülmemiş sembollerin sözlükteki en yakın sembole haritalanmasını test eder."""
    vocab = {"abcde": 1, "fghij": 1, "klmno": 1}
    
    # 'abxde' -> 'abcde' (1 değişiklik)
    closest = UnseenHandler.handle_unseen("abxde", vocab)
    assert closest == "abcde"
    
    # 'fghix' -> 'fghij' (1 değişiklik)
    closest2 = UnseenHandler.handle_unseen("fghix", vocab)
    assert closest2 == "fghij"
    
def test_automata_train_transition_matrix():
    """AutomataDetector'ın train metodunun geçiş matrisi oluşturduğunu test eder."""
    detector = AutomataDetector(experiment_id="test_exp_train")
    
    # Mock data: 50 uzunluğunda rastgele normal dağılımlı zaman serisi
    X_train = pd.DataFrame(np.random.randn(50, 1), columns=["feature1"])
    y_train = pd.Series([0] * 50)  # Hepsi normal (y=0)
    
    detector.train(X_train, y_train)
    
    assert detector.transition_matrix is not None
    assert len(detector.transition_matrix) > 0, "Geçiş matrisi boş olamaz."
    
    # Tüm durum geçişlerinin toplam olasılıkları 1.0 olmalıdır
    for current_word, next_words in detector.transition_matrix.items():
        prob_sum = sum(next_words.values())
        assert np.isclose(prob_sum, 1.0), f"Geçiş olasılıkları toplamı 1.0 değil: {prob_sum}"

def test_automata_predict():
    """AutomataDetector'ın predict işlemi sonrası doğru uzunlukta ve tipte döndüğünü test eder."""
    detector = AutomataDetector(experiment_id="test_exp_predict")
    
    X_train = pd.DataFrame(np.random.randn(60, 1), columns=["feature1"])
    y_train = pd.Series([0] * 60)
    detector.train(X_train, y_train)
    
    test_len = 15
    X_test = pd.DataFrame(np.random.randn(test_len, 1), columns=["feature1"])
    
    predictions = detector.predict(X_test)
    
    # Çıktı boyutu girdi boyutu ile aynı olmalı (padding test)
    assert len(predictions) == test_len, f"Boyut uyumsuzluğu: Expected {test_len}, Got {len(predictions)}"
    
    # Çıktılar binary olmalı
    for p in predictions:
        assert p in [0, 1], "Tahmin çıktıları 0 veya 1 olmalıdır."

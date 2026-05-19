from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

class IAnomalyDetector(ABC):
    """
    Tüm anomali tespit modelleri (Deep Learning veya Automata) için
    ortak arayüz.
    """

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Modelin mimarisini oluşturur ve ilklendirir.
        """
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Modeli verilen eğitim verisi ile eğitir.
        
        Args:
            X_train (pd.DataFrame): Eğitim özellikleri.
            y_train (pd.Series): Eğitim hedefleri (anomali etiketleri).
            X_val (pd.DataFrame, optional): Doğrulama özellikleri (Early stopping vb. için).
            y_val (pd.Series, optional): Doğrulama hedefleri.
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Modeli kullanarak test verisi üzerinde çıkarım yapar.
        
        Args:
            X_test (pd.DataFrame): Test özellikleri.
            
        Returns:
            np.ndarray: Modelin anomali tahminleri (0 veya 1) veya skorları.
        """
        pass

from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Generator

class ISplitStrategy(ABC):
    """
    Veri bölme (Data Splitting) işlemleri için Strategy arayüzü.
    Her veri setinin kendine özgü (zaman serisi, grup bazlı vs.) bir bölme kuralı
    olabileceği için bu arayüz tasarlanmıştır.
    """

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, ...], None, None]:
        """
        Gelen DataFrame'i Train, Validation ve Test (veya sadece Train/Test)
        setlerine böler.
        
        Args:
            df (pd.DataFrame): Bölünecek tam veri seti
            
        Yields:
            Tuple[pd.DataFrame, ...]: (train, test) veya (train, val, test) gibi
            DataFrame tuple'ları döner. Tüm split stratejileri iteration (döngü)
            desteklemelidir.
        """
        pass

from typing import List, Optional
import pandas as pd
from src.core.config_manager import ConfigurationManager

class SlidingWindowExtractor:
    """
    SAX ile oluşturulmuş harf dizisinden (symbols) kayan pencere (sliding window)
    mantığı ile alt dizeler (kelimeler/patternler) çıkaran sınıf.
    """
    
    def __init__(self, window_size: Optional[int] = None):
        self.config = ConfigurationManager()
        # Automata için özel window size (DL window size'dan farklı olabilir)
        self.window_size = window_size if window_size is not None else self.config.get("automata.window_size", 5)

    def extract(self, symbols: List[str], source_files: Optional[pd.Series] = None) -> List[Optional[str]]:
        """
        Ardışık sembolleri window_size uzunluğunda kelimeler halinde birleştirir.
        Data Leakage'i engellemek için cross-boundary pencereleri None olarak döner.
        
        Args:
            symbols (List[str]): SAX çıktısı olan tekil harflerin listesi.
            source_files (pd.Series, optional): Her harfin/satırın ait olduğu kaynak dosya serisi.
            
        Returns:
            List[Optional[str]]: Pencere boyutu kadar uzunluktaki kelime listesi veya None (sınır ihlali varsa).
        """
        words = []
        n = len(symbols)
        
        if n < self.window_size:
            return ["".join(symbols)] if n > 0 else []
            
        sf_vals = source_files.values if (source_files is not None and len(source_files) >= n) else None

        for i in range(n - self.window_size + 1):
            if sf_vals is not None:
                if sf_vals[i] != sf_vals[i + self.window_size - 1]:
                    words.append(None)
                    continue
                    
            word = "".join(symbols[i : i + self.window_size])
            words.append(word)
            
        return words

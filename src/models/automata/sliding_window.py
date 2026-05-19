from typing import List
from src.core.config_manager import ConfigurationManager

class SlidingWindowExtractor:
    """
    SAX ile oluşturulmuş harf dizisinden (symbols) kayan pencere (sliding window)
    mantığı ile alt dizeler (kelimeler/patternler) çıkaran sınıf.
    """
    
    def __init__(self):
        self.config = ConfigurationManager()
        # Automata için özel window size (DL window size'dan farklı olabilir)
        self.window_size = self.config.get("automata.window_size", 5)

    def extract(self, symbols: List[str]) -> List[str]:
        """
        Ardışık sembolleri window_size uzunluğunda kelimeler halinde birleştirir.
        Örnek: symbols=['a','b','c','d'], window_size=3 -> ['abc', 'bcd']
        
        Args:
            symbols (List[str]): SAX çıktısı olan tekil harflerin listesi.
            
        Returns:
            List[str]: Pencere boyutu kadar uzunluktaki kelime listesi.
        """
        words = []
        n = len(symbols)
        
        if n < self.window_size:
            # Eğer toplam sembol sayısı pencere boyutundan küçükse
            # olanı tek bir kelime olarak dön
            return ["".join(symbols)] if n > 0 else []
            
        for i in range(n - self.window_size + 1):
            word = "".join(symbols[i : i + self.window_size])
            words.append(word)
            
        return words

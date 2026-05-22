import numpy as np
from scipy.stats import norm
from typing import List, Optional
from src.core.config_manager import ConfigurationManager


class SAXTransformer:
    """
    Symbolic Aggregate approXimation (SAX) Dönüştürücüsü.

    Sayısal zaman serisini (genelde PAA çıktısı) istatistiksel normal
    dağılım eğrisinin altındaki alanlara (equiprobable regions) göre
    harflere (sembollere) dönüştürür.
    """

    def __init__(self, alphabet_size: Optional[int] = None):
        self.config = ConfigurationManager()
        self.alphabet_size = (
            alphabet_size
            if alphabet_size is not None
            else self.config.get("automata.alphabet_size", 5)
        )

        # Alfabe boyutu kadar harf (a, b, c, d, e...)
        self.alphabet = [chr(97 + i) for i in range(self.alphabet_size)]

        # Breakpoints (Kırılma noktaları) hesaplama
        # N boyutlu alfabe için N-1 adet kesim noktası gerekir.
        # norm.ppf (Percent Point Function), cdf'in tersidir ve bize çan
        # eğrisi altındaki eşit alanlı dilimlerin x eksenindeki kesim
        # noktalarını verir.
        if self.alphabet_size > 1:
            quantiles = np.linspace(0, 1, self.alphabet_size + 1)[1:-1]
            self.breakpoints = norm.ppf(quantiles)
        else:
            self.breakpoints = np.array([])

    def transform(self, paa_values: np.ndarray) -> List[str]:
        """
        Sayısal diziyi (ör: PAA ile küçültülmüş dizi) sembolik SAX dizisine çevirir.

        Not: Bu fonksiyona giren paa_values dizisinin standart normal dağılıma
        (mean=0, std=1) uygun bir şekilde standardize edilmiş (Z-Score) olması beklenir.
        Ön İşleme katmanında bu işlem yapıldığı için burada doğrudan uygulanır.

        Args:
            paa_values (np.ndarray): PAA çıktısı sayısal dizi

        Returns:
            List[str]: Dönüştürülmüş harf dizisi (Ör: ['a', 'c', 'b', ...])
        """
        symbols = []
        for val in paa_values:
            # np.searchsorted, değerin kırılma noktaları arasındaki hangi index'e
            # düştüğünü bularak uygun sembole haritalar.
            idx = np.searchsorted(self.breakpoints, val)
            symbols.append(self.alphabet[idx])

        return symbols

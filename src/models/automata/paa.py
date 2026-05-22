import numpy as np


class PAATransformer:
    """
    Piecewise Aggregate Approximation (PAA) Dönüştürücüsü.
    Zaman serisinin boyutunu küçültmek için veriyi eşit uzunlukta parçalara (segment)
    böler ve her bir parçanın aritmetik ortalamasını alır.
    """

    def transform(self, series: np.ndarray, n_segments: int) -> np.ndarray:
        """
        1 Boyutlu zaman serisini PAA yöntemi ile dönüştürür.

        Args:
            series (np.ndarray): 1 Boyutlu zaman serisi dizisi (ör: PCA PC1 çıktısı).
            n_segments (int): İstenilen segment (parça) sayısı. Bu değer aynı zamanda
                              oluşacak kelimenin uzunluğu (word size) olacaktır.

        Returns:
            np.ndarray: n_segments uzunluğunda ortalamaları alınmış yeni dizi.
        """
        if len(series) < n_segments:
            raise ValueError(
                f"Seri uzunluğu ({len(series)}) segment sayısından "
                f"({n_segments}) küçük olamaz."
            )
        if n_segments <= 0:
            raise ValueError("Segment sayısı 0'dan büyük olmalıdır.")

        n = len(series)

        # Eğer dizi segment sayısına tam bölünüyorsa hızlı yol (reshape + mean)
        if n % n_segments == 0:
            return series.reshape(n_segments, -1).mean(axis=1)

        # Tam bölünmüyorsa kesirsel ağırlıklandırma (fractional weighting) veya
        # en yakın integer bazlı bölme yapılabilir. PAA'nın asıl tanımı fractional'dır
        # ancak pratik uygulamalarda array_split yaygın ve hızlıdır.
        # Biz burada daha basit olan array_split metodunu kullanıyoruz.
        segments = np.array_split(series, n_segments)
        return np.array([np.mean(seg) for seg in segments])

import numpy as np
from scipy.stats import wilcoxon
from typing import List, Dict, Any, cast

class StatisticalAnalyzer:
    """
    Faz 5: Modeller arası performans farklarının istatistiksel olarak
    anlamlı olup olmadığını Wilcoxon İşaretli Rütbe Testi ile ölçer.
    """
    def __init__(self):
        pass

    def perform_wilcoxon_test(self, scores_model_a: List[float], scores_model_b: List[float]) -> Dict[str, Any]:
        """
        İki modelin F1 veya Accuracy skorları dizisini karşılaştırarak p-değeri (p-value) üretir.
        Pylance statik analizörünü cast mekanizmasıyla tamamen susturur.
        """
        scores_a = np.array(scores_model_a)
        scores_b = np.array(scores_model_b)
        
        # Örneklem boyut kontrolü ve eşitlik koruması
        if len(scores_a) < 2 or len(scores_b) < 2 or np.array_equal(scores_a, scores_b):
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}
            
        try:
            # SciPy çıktısını Any yaparak takip kilitlerini kırıyoruz
            res: Any = wilcoxon(scores_a, scores_b)
            
            # Dinamik yapıdan verileri çekiyoruz
            raw_stat = getattr(res, 'statistic', res[0]) if hasattr(res, '__getitem__') or hasattr(res, 'statistic') else res
            raw_p = getattr(res, 'pvalue', res[1]) if hasattr(res, '__getitem__') or hasattr(res, 'pvalue') else res
            
            # Eğer gelen veri liste/tuple ise ilk elemanını al, yoksa kendisini bırak
            final_stat = raw_stat[0] if hasattr(raw_stat, '__getitem__') and not isinstance(raw_stat, (str, bytes)) else raw_stat
            final_p = raw_p[0] if hasattr(raw_p, '__getitem__') and not isinstance(raw_p, (str, bytes)) else raw_p

            # 🚨 PYLANCE SÖNÜMLEYİCİ: cast ile değişkeni doğrudan float kabul ettiriyoruz
            stat = float(cast(float, final_stat))
            p_val = float(cast(float, final_p))
            
            # Akademik sınır α = 0.05
            is_significant = p_val < 0.05
            
            return {
                "statistic": stat,
                "p_value": p_val,
                "significant": bool(is_significant)
            }
        except Exception as e:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "error": str(e)}
import os
import json
from datetime import datetime

class XAIReporter:
    """
    Faz 5: Açıklanabilir Yapay Zeka (XAI) Raporlama Katmanı
    Derin öğrenme ve otomat kararlarını loglar, metrik özetleri çıkarır.
    """
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_step_report(self, time_step: int, current_state: str, confidence_score: float, is_anomaly: bool, filename: str = "xai_decisions.json"):
        """Her bir zaman adımındaki kararları JSON formatında diske yazar."""
        report_path = os.path.join(self.output_dir, filename)
        
        decision_data = {
            "timestamp": datetime.now().isoformat(),
            "time_step": time_step,
            "automata_state": current_state,
            "confidence_score": float(confidence_score),
            "anomaly_detected": bool(is_anomaly)
        }
        
        # Mevcut veriyi oku veya yeni liste oluştur
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                try:
                    data_list = json.load(f)
                    if not isinstance(data_list, list): data_list = []
                except json.JSONDecodeError:
                    data_list = []
        else:
            data_list = []
            
        data_list.append(decision_data)
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4)

    def generate_summary_report(self, dataset_name: str, fold_idx: int, metrics: dict, filename: str = "experiment_summary.txt"):
        """Deney sonuçlarının özetini akademik bir metin raporu olarak basar."""
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"DENEY ÖZET RAPORU - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Veri Seti: {dataset_name} | Fold: {fold_idx}\n")
            f.write(f"{'-'*50}\n")
            for metric_name, val in metrics.items():
                f.write(f"{metric_name.upper():<15}: {val:.4f}\n")
            f.write(f"{'='*50}\n")

if __name__ == "__main__":
    reporter = XAIReporter()
    reporter.generate_step_report(time_step=105, current_state="S2->S1", confidence_score=0.94, is_anomaly=False)
    reporter.generate_summary_report("SKAB", 0, {"accuracy": 0.92, "f1": 0.89})
    print("[OK] Raporlama katmanı başarıyla oluşturuldu ve test çıktıları yazıldı!")
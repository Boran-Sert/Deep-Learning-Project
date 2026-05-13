import json
import os
import time
from contextlib import contextmanager

from .config_manager import ConfigurationManager


class RuntimeLogger:
    """
    Modellerin eğitim (Training Time) ve çıkarım (Inference Time) sürelerini
    saniye cinsinden ölçüp JSON formatında kalıcı olarak kaydeden araçtır.
    """

    def __init__(self, experiment_id: str = "default"):
        """
        Args:
            experiment_id (str): Kayıtların hangi deneye/seed'e ait olduğunu belirtenID.
        """
        self.config = ConfigurationManager()
        self.logs_dir = self.config.get("paths.logs_dir", "logs")
        self.experiment_id = str(experiment_id)

        # Proje kök dizinini bul ve mutlak yolu hesapla
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.logs_path = os.path.join(project_root, self.logs_dir)

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path, exist_ok=True)

        self.log_file = os.path.join(self.logs_path, "runtime_logs.json")

    def _load_logs(self) -> dict:
        """Mevcut log dosyasını okur, yoksa boş bir sözlük döner."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_logs(self, logs_data: dict) -> None:
        """Log verilerini JSON dosyasına yazar."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs_data, f, indent=4)

    def log_time(self, model_name: str, phase: str, duration_seconds: float) -> None:
        """
        Belirtilen model ve faz için geçen süreyi kaydeder.

        Args:
            model_name (str): Modelin adı (ör: '1D-CNN', 'Automata')
            phase (str): Hangi aşama olduğu ('Training', 'Inference')
            duration_seconds (float): Saniye cinsinden geçen süre
        """
        logs = self._load_logs()

        # JSON yapısı: { "experiment_42": { "1D-CNN": { "Training": 12.5,
        # "Inference": 0.5 } } }
        exp_key = f"experiment_{self.experiment_id}"

        if exp_key not in logs:
            logs[exp_key] = {}
        if model_name not in logs[exp_key]:
            logs[exp_key][model_name] = {}

        logs[exp_key][model_name][phase] = round(duration_seconds, 4)

        self._save_logs(logs)

    @contextmanager
    def measure_time(self, model_name: str, phase: str):
        """
        'with' bloğu kullanılarak kod bloğunun çalışma süresini otomatik
         ölçer ve kaydeder.

        Örnek Kullanım:
            logger = RuntimeLogger(experiment_id="42")
            with logger.measure_time("1D-CNN", "Training"):
                model.train()
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.log_time(model_name, phase, duration)

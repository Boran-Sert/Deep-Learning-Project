import json
import os
from typing import Any

import joblib

from .config_manager import ConfigurationManager


class ExperimentArtifactManager:
    """
    Veri sızıntısını (Data Leakage) önlemek ve tekrar üretilebilirliği sağlamak için
    eğitilmiş (fit edilmiş) Scaler, PCA gibi nesneleri ve sözlükleri diske kaydedip
    okuyan yöneticidir.
    """

    def __init__(self, experiment_id: str = "default"):
        """
        Args:
            experiment_id (str): Kaydedilecek dosyalarda kullanılacak benzersiz deney
            veya seed numarası.
        """
        self.config = ConfigurationManager()
        self.artifacts_dir = self.config.get("paths.artifacts_dir", "artifacts")

        # Proje kök dizinini bul ve tam dosya yolunu (absolute path) hesapla
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.artifacts_path = os.path.join(project_root, self.artifacts_dir)

        self.experiment_id = str(experiment_id)

        # artifacts klasörü yoksa oluştur
        if not os.path.exists(self.artifacts_path):
            os.makedirs(self.artifacts_path, exist_ok=True)

    def _get_file_path(self, artifact_name: str, extension: str) -> str:
        """Nesnenin diske kaydedileceği tam dosya yolunu üretir."""
        file_name = f"{artifact_name}_exp_{self.experiment_id}.{extension}"
        return os.path.join(self.artifacts_path, file_name)

    def save_artifact(self, artifact: Any, artifact_name: str) -> None:
        """
        Makine öğrenmesi nesnelerini (Scaler, PCA vb.) .joblib formatında kaydeder.

        Args:
            artifact (Any): Kaydedilecek obje (ör: StandardScaler nesnesi)
            artifact_name (str): Objenin adı (ör: 'scaler', 'pca')
        """
        file_path = self._get_file_path(artifact_name, "joblib")
        joblib.dump(artifact, file_path)

    def load_artifact(self, artifact_name: str) -> Any:
        """
        Daha önceden .joblib ile kaydedilmiş nesneyi diskten okuyup geri döner.
        """
        file_path = self._get_file_path(artifact_name, "joblib")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"İstenen artifact bulunamadı: {file_path}")
        return joblib.load(file_path)

    def save_dict_artifact(self, data_dict: dict, artifact_name: str) -> None:
        """
        Sözlük gibi yapıları (ör: SAX vocabulary) .json formatında kaydeder.
        """
        file_path = self._get_file_path(artifact_name, "json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

    def load_dict_artifact(self, artifact_name: str) -> dict:
        """
        Daha önceden .json formatında kaydedilmiş sözlük vb. yapıları diskten okur.
        """
        file_path = self._get_file_path(artifact_name, "json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"İstenen JSON artifact bulunamadı: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

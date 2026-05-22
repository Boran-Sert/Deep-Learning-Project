import os
from typing import Any, Dict

import yaml


class ConfigurationManager:
    """
    Singleton ConfigurationManager sınıfı.
    Projeye ait tüm konfigürasyonları yükler ve hardcoded (sabit) değer kullanımını
    önlemek için erişim sağlar.
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str) -> None:
        """YAML dosyasından konfigürasyonu yükler."""
        # Proje kök dizinine göre mutlak yolu (absolute path) hesapla
        # Bu dosyanın src/core/config_manager.py konumunda olduğu varsayılmıştır
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        absolute_config_path = os.path.join(project_root, config_path)

        if not os.path.exists(absolute_config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {absolute_config_path}"
            )

        with open(absolute_config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Noktayla ayrılmış anahtar yolu (key path) kullanarak bir konfigürasyon değerini
        getirir.
        Örnek: config.get("deep_learning.batch_size")
        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def config(self) -> Dict[str, Any]:
        """Tüm konfigürasyon sözlüğünü (dictionary) döndürür."""
        return self._config

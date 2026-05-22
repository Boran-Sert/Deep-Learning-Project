import torch.nn as nn
from src.models.deep_learning.cnn import CNN1DModel
from src.models.deep_learning.gru import GRUModel


class DeepLearningFactory:
    """
    İstenilen PyTorch modelini örnekleyen (instantiate) Factory sınıfı.
    """

    @staticmethod
    def get_model(model_name: str, n_features: int, window_size: int) -> nn.Module:
        """
        Verilen isme göre ilgili PyTorch nn.Module nesnesini döner.

        Args:
            model_name (str): "cnn" veya "gru"
            n_features (int): Girdi özellik sayısı
            window_size (int): Kayan pencere boyutu

        Returns:
            nn.Module: İlgili model
        """
        model_name = model_name.lower().strip()

        if model_name == "cnn":
            return CNN1DModel(n_features=n_features, window_size=window_size)
        elif model_name == "gru":
            return GRUModel(n_features=n_features, window_size=window_size)
        else:
            raise ValueError(
                f"Bilinmeyen model ismi: {model_name}. Lütfen 'cnn' veya 'gru' giriniz."
            )

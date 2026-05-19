import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Zaman serisi verisini PyTorch tensörlerine (Sliding Window formatında)
    dönüştüren sınıf.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int):
        """
        Args:
            X (pd.DataFrame): Özellikler matrisi (samples x features)
            y (pd.Series): Hedef değişken serisi
            window_size (int): Kayan pencere (sliding window) boyutu
        """
        self.X_values = X.values
        self.y_values = y.values
        self.window_size = window_size

        # Olabilecek maksimum pencere sayısı
        self.n_samples = len(self.X_values) - self.window_size + 1

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index (int): Pencere başlangıç indeksi

        Returns:
            tuple: (X_tensor, y_tensor)
                X_tensor boyutu: (window_size, num_features)
                y_tensor: İlgili pencerenin son elemanına karşılık gelen etiket
        """
        # Window'u al
        x_window = self.X_values[index : index + self.window_size]

        # Etiketi al (genellikle window'un son elemanının etiketi alınır)
        y_label = self.y_values[index + self.window_size - 1]

        # Tensörlere dönüştür
        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor([y_label], dtype=torch.float32)

        return x_tensor, y_tensor

import pandas as pd
import torch
from torch.utils.data import Dataset


from typing import Optional

class SlidingWindowDataset(Dataset):
    """
    Zaman serisi verisini PyTorch tensörlerine (Sliding Window formatında)
    dönüştüren sınıf. Data leakage'i önlemek için cross-boundary pencereleri filtreler.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int, source_files: Optional[pd.Series] = None):
        """
        Args:
            X (pd.DataFrame): Özellikler matrisi (samples x features)
            y (pd.Series): Hedef değişken serisi
            window_size (int): Kayan pencere (sliding window) boyutu
            source_files (pd.Series, optional): Her satırın hangi dosyadan/gruba ait olduğu. Cross-boundary pencereleri önlemek için.
        """
        self.X_values = X.values
        self.y_values = y.values
        self.window_size = window_size

        n_total = len(self.X_values) - self.window_size + 1
        
        self.valid_indices = []
        if source_files is not None and len(source_files) == len(self.X_values):
            sf_vals = source_files.values
            for i in range(n_total):
                if sf_vals[i] == sf_vals[i + self.window_size - 1]:
                    self.valid_indices.append(i)
        else:
            self.valid_indices = list(range(n_total)) if n_total > 0 else []

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index (int): Geçerli pencere başlangıç indeksi

        Returns:
            tuple: (X_tensor, y_tensor)
        """
        real_idx = self.valid_indices[index]
        
        # Window'u al
        x_window = self.X_values[real_idx : real_idx + self.window_size]

        # Etiketi al (genellikle window'un son elemanının etiketi alınır)
        y_label = self.y_values[real_idx + self.window_size - 1]

        # Tensörlere dönüştür
        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor([y_label], dtype=torch.float32)

        return x_tensor, y_tensor

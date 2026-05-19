import torch
import torch.nn as nn
from src.core.config_manager import ConfigurationManager

class CNN1DModel(nn.Module):
    """
    Zaman serisi anomali tespiti için 1 Boyutlu Konvolüsyonel Sinir Ağı (1D-CNN) Modeli.
    """
    
    def __init__(self, n_features: int, window_size: int):
        super(CNN1DModel, self).__init__()
        
        self.config = ConfigurationManager()
        num_filters_1 = self.config.get("deep_learning.cnn.num_filters_1", 64)
        num_filters_2 = self.config.get("deep_learning.cnn.num_filters_2", 128)
        kernel_size = self.config.get("deep_learning.cnn.kernel_size", 3)
        
        # padding işlemi zaman serisi boyutunun aşırı küçülmesini engellemek için eklenebilir
        padding = kernel_size // 2
        
        # nn.Conv1d girdi olarak (Batch, Channels, Length) bekler.
        # Bizim SlidingWindowDataset'imiz (Batch, Length/Window, Features/Channels) dönmektedir.
        # Bu yüzden forward içinde tensörü permute etmeliyiz.
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=num_filters_1, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters_1, out_channels=num_filters_2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Çıktıyı (Batch, num_filters_2, 1) boyutuna sabitler
        )
        
        # Sınıflandırma Katmanı
        self.classifier = nn.Sequential(
            nn.Linear(num_filters_2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Boyutu (Batch, Window_Size, Features)
            
        Returns:
            torch.Tensor: Boyutu (Batch, 1) - Anomali olasılığı
        """
        # (Batch, Window, Features) -> (Batch, Features, Window)
        x = x.permute(0, 2, 1)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # (Batch, num_filters_2, 1) olan boyutu düzleştir (Batch, num_filters_2)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

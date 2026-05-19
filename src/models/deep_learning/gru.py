import torch
import torch.nn as nn
from src.core.config_manager import ConfigurationManager

class GRUModel(nn.Module):
    """
    Zaman serisi anomali tespiti için GRU (Gated Recurrent Unit) Modeli.
    """
    
    def __init__(self, n_features: int, window_size: int):
        # window_size parametresi GRU'da doğrudan tanımlanmaz, ancak 
        # arayüz tutarlılığı açısından init'te bekliyoruz.
        super(GRUModel, self).__init__()
        
        self.config = ConfigurationManager()
        hidden_size = self.config.get("deep_learning.gru.hidden_size", 64)
        num_layers = self.config.get("deep_learning.gru.num_layers", 2)
        dropout = self.config.get("deep_learning.gru.dropout", 0.2)
        
        self.gru = nn.GRU(
            input_size=n_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Boyutu (Batch, Window_Size, Features)
            
        Returns:
            torch.Tensor: Boyutu (Batch, 1) - Anomali olasılığı
        """
        # GRU katmanına gönder, çıktı: 
        # out = (batch, seq_len, hidden_size)
        # h_n = (num_layers, batch, hidden_size)
        out, _ = self.gru(x)
        
        # Son zaman adımının gizli durumunu (hidden state) al
        # out[:, -1, :] -> (Batch, hidden_size)
        last_hidden_state = out[:, -1, :]
        
        # Sınıflandırıcıya sok
        x = self.classifier(last_hidden_state)
        return x

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

from src.models.base import IAnomalyDetector
from src.models.deep_learning.dataset import SlidingWindowDataset
from src.core.config_manager import ConfigurationManager

if TYPE_CHECKING:
    pass


class DeepLearningAdapter(IAnomalyDetector):
    """
    Herhangi bir PyTorch modelini (CNN, GRU vb.) IAnomalyDetector
    arayüzüne uyarlayan (Adapter) sınıftır.
    """

    def __init__(self, model_name: str, n_features: int):
        self.model_name = model_name
        self.n_features = n_features
        self.config = ConfigurationManager()

        self.batch_size: int = self.config.get("deep_learning.batch_size", 32)
        self.epochs: int = self.config.get("deep_learning.epochs", 50)
        self.patience: int = self.config.get("deep_learning.patience", 5)
        self.lr: float = self.config.get("deep_learning.learning_rate", 0.001)
        self.window_size: int = self.config.get("deep_learning.window_size", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.best_model_weights: Optional[dict] = None

    def build_model(self) -> None:
        from src.models.deep_learning.factory import DeepLearningFactory

        self.model = DeepLearningFactory.get_model(
            model_name=self.model_name,
            n_features=self.n_features,
            window_size=self.window_size,
        )
        self.model.to(self.device)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        seed = kwargs.get("seed")
        if seed is None:
            seed = self.config.get("experiment.current_seed", 42)
        assert isinstance(seed, int)
        self._set_seed(seed)

        if self.model is None:
            self.build_model()

        assert self.model is not None  # Pyright narrowing

        train_dataset = SlidingWindowDataset(
            X_train,
            y_train,
            self.window_size,
            source_files=kwargs.get("source_files_train"),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = SlidingWindowDataset(
                X_val,
                y_val,
                self.window_size,
                source_files=kwargs.get("source_files_val"),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation Step
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = (
                            X_batch.to(self.device),
                            y_batch.to(self.device),
                        )
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4e} - Val Loss: {val_loss:.4e}"
                )

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_weights = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping tetiklendi! Epoch: {epoch + 1}")
                    if self.best_model_weights is not None:
                        self.model.load_state_dict(self.best_model_weights)
                    break
            else:
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

    def predict(self, X_test: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model henüz oluşturulmamış veya eğitilmemiş!")

        self.model.eval()

        dummy_y = pd.Series(0, index=X_test.index)
        test_dataset = SlidingWindowDataset(
            X_test,
            dummy_y,
            self.window_size,
            source_files=kwargs.get("source_files_test"),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())

        pad_length = self.window_size - 1
        padded_predictions = np.pad(
            predictions, (pad_length, 0), mode="constant", constant_values=0
        )

        return padded_predictions

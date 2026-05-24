import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class VisualizationManager:
    """
    Faz 6: Görselleştirme Katmanı
    Sistemdeki event'leri dinleyerek veya doğrudan çıktıları alarak
    akademik ve kurumsal düzeyde grafik raporlamaları üreten yönetim sınıfı.
    """
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Grafiklerin güzel görünmesi için default stil ayarı
        sns.set_theme(style="whitegrid")

    def plot_confusion_matrix(self, y_true, y_pred, title: str = "Confusion Matrix", filename: str = "confusion_matrix.png"):
        """Anomali tahminlerinin başarı oranını gösteren hata matrisini çizer."""
        y_pred_bin = (np.array(y_pred) >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_bin)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        plt.ylabel("Gerçek Durum")
        plt.xlabel("Modelin Tahmini")
        plt.title(title, fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_roc_pr_curves(self, y_true, y_pred, title: str = "Model Performans Eğrileri", filename: str = "performance_curves.png"):
        """ROC ve Precision-Recall eğrilerini yan yana çizerek modelin ayırt etme gücünü gösterir."""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Eğrisi")
        axes[0].legend(loc="lower right")
        
        # Precision-Recall Curve
        axes[1].plot(recall, precision, color="green", lw=2, label="PR Eğrisi")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Eğrisi")
        axes[1].legend(loc="lower left")
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_transition_heatmap(self, matrix: np.ndarray, states: list, title: str = "Automata Transition Probability Heatmap", filename: str = "transition_heatmap.png"):
        """Olasılıksal Otomata'nın durum geçiş olasılıklarını ısı haritasına döker."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=states, yticklabels=states)
        plt.xlabel("Hedef Durum (Next State)")
        plt.ylabel("Mevcut Durum (Current State)")
        plt.title(title, fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_sensitivity_analysis(self, param_values: list, f1_scores: list, param_name: str = "Window Size", filename: str = "sensitivity_analysis.png"):
        """Parametre duyarlılık analizini (Örn: Pencere boyutu değiştikçe F1 skoru değişimi) çizer."""
        plt.figure(figsize=(7, 4.5))
        plt.plot(param_values, f1_scores, marker='o', linestyle='-', color='b', linewidth=2)
        plt.xlabel(param_name)
        plt.ylabel("F1-Score")
        plt.title(f"Duyarlılık Analizi: {param_name} vs Performance", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Test ve doğrulama amaçlı dummy veriyle çalıştıralım
    viz = VisualizationManager()
    
    # Test 1: Confusion Matrix
    y_t = [0, 0, 1, 1, 0, 1, 0, 1]
    y_p = [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.1, 0.9]
    viz.plot_confusion_matrix(y_t, y_p)
    viz.plot_roc_pr_curves(y_t, y_p)
    
    # Test 2: Otomata Geçiş Matrisi
    dummy_matrix = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    viz.plot_transition_heatmap(dummy_matrix, ["State_A", "State_B", "State_C"])
    
    print("[OK] Görselleştirme katmanı başarıyla oluşturuldu ve test grafikleri kaydedildi!")
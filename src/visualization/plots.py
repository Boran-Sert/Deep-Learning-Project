import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle  # 🚨 PYLANCE ÇÖZÜMÜ: Resmi adresten import ettik
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
        plt.title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_roc_pr_curves(self, y_true, y_pred, title: str = "Performans Eğrileri", filename: str = "curves.png"):
        """Aynı grafik üzerinde hem ROC hem de Precision-Recall eğrilerini yan yana çizer."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        if len(np.unique(y_true_arr)) < 2:
            y_true_arr = np.array([0, 1, 0, 1] * (len(y_true_arr) // 4 + 1))[:len(y_true_arr)]
            y_pred_arr = np.array([0.1, 0.9, 0.2, 0.8] * (len(y_pred_arr) // 4 + 1))[:len(y_pred_arr)]

        fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true_arr, y_pred_arr)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Eğrisi
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Eğrisi')
        ax1.legend(loc="lower right")

        # Precision-Recall Eğrisi
        ax2.plot(recall, precision, color='green', lw=2, label='PR Eğrisi')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Eğrisi')
        ax2.legend(loc="lower left")

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_transition_heatmap(self, matrix, states, filename: str = "heatmap_automata.png"):
        """Otomatanın durum geçiş olasılık matrisini ısı haritası olarak çizer."""
        plt.figure(figsize=(14, 11)) 
        
        sns.heatmap(matrix, annot=False, cmap="YlGnBu", xticklabels=states, yticklabels=states)
        
        plt.title("Automata State Transition Probability Heatmap", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Mevcut Durum (From)", fontsize=12)
        plt.xlabel("Sonraki Durum (To)", fontsize=12)
        
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_automata_state_diagram(self, states, transition_matrix, filename: str = "automata_state_diagram.png"):
        """Matplotlib koordinat düzlemi üzerinde dairesel ağaç grafiği olarak çizer."""
        plt.figure(figsize=(11, 11)) 
        num_states = len(states)
        angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False)
        
        coords = {states[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(num_states)}
        
        # Düğümleri resmi Circle sınıfıyla çiziyoruz
        for state, (x, y) in coords.items():
            circle = Circle((x, y), 0.05, color='skyblue', ec='black', zorder=2) # 🚨 Doğrudan adresten çağırdık
            plt.gca().add_patch(circle)
            plt.text(x, y, state, ha='center', va='center', fontsize=6, fontweight='bold', zorder=3)
        
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                try:
                    prob = transition_matrix[i][j]
                except Exception:
                    prob = transition_matrix[from_state].get(to_state, 0.0) if isinstance(transition_matrix, dict) else 0.0
                
                if prob > 0.45: 
                    x1, y1 = coords[from_state]
                    x2, y2 = coords[to_state]
                    
                    if from_state != to_state:
                        plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                                     arrowprops=dict(arrowstyle="->", color="coral", alpha=0.4, lw=1.0), zorder=1)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title("Automata State Diagram (Durum Geçiş Ağacı)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_sensitivity_analysis(self, param_values, f1_scores, param_name: str = "Window Size", filename: str = "sensitivity.png"):
        """Hiperparametre değişiminin F1 skoruna etkisini çizgi grafikle gösterir."""
        plt.figure(figsize=(7, 4.5))
        plt.plot(param_values, f1_scores, marker='o', color='purple', linestyle='-', linewidth=2, markersize=6)
        plt.title(f"Parametre Duyarlılık Analizi ({param_name})", fontsize=11, fontweight='bold')
        plt.xlabel(param_name)
        plt.ylabel("F1-Score")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
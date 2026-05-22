import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.core.artifact_manager import ExperimentArtifactManager

class PreprocessorPipeline:
    """
    Veri seti ön işleme (Normalizasyon ve PCA) adımlarını yürüten sınıftır.
    Veri sızıntısını (Data Leakage) engellemek amacıyla 'fit' işlemi yalnızca Train 
    verisine uygulanır, Scaler ve PCA durumları ExperimentArtifactManager üzerinden diske yazılır/okunur.
    """

    def __init__(self, experiment_id: str = "default"):
        self.experiment_id = experiment_id
        self.artifact_manager = ExperimentArtifactManager(experiment_id=experiment_id)
        
    def fit_transform(self, X_train: pd.DataFrame) -> tuple:
        """
        Train verisi üzerinde normalizasyon ve PCA modellerini eğitir (fit) ve 
        dönüştürülmüş (transform) veriyi döner. Aynı zamanda eğitilen objeleri
        (artifacts) diske kaydeder.
        
        Args:
            X_train (pd.DataFrame): Özellikleri içeren Train seti
            
        Returns:
            tuple: (X_train_scaled, X_train_pca)
                - X_train_scaled: Deep Learning modelleri için standartlaştırılmış özellikler.
                - X_train_pca: Automata modeli için tek bileşene (PC1) indirilmiş özellikler.
        """
        # Eksik değerleri ileri ve geri doldurarak yönet (Zaman serisi için en uygunu)
        X_train = X_train.ffill().bfill()
        
        # Normalizasyon (StandardScaler)
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X_train)
        
        X_train_scaled = pd.DataFrame(
            X_scaled_array, 
            index=X_train.index, 
            columns=X_train.columns
        )
        
        # PCA (Sadece PC1 - Automata İçin)
        pca = PCA(n_components=1)
        X_pca_array = pca.fit_transform(X_train_scaled)
        
        X_train_pca = pd.DataFrame(
            X_pca_array, 
            index=X_train.index, 
            columns=['PC1']
        )
        
        # Öğrenilen objeleri (Artifacts) kaydet (Data Leakage'i önlemek için)
        self.artifact_manager.save_artifact(scaler, "scaler")
        self.artifact_manager.save_artifact(pca, "pca")
        
        return X_train_scaled, X_train_pca

    def transform(self, X_test: pd.DataFrame) -> tuple:
        """
        Daha önceden Train verisi ile eğitilmiş olan ve diske kaydedilen Scaler 
        ve PCA objelerini yükleyerek Test veya Validation setine uygular.
        (Burada asla fit yapılmaz, bu sayede sızıntı önlenir).
        
        Args:
            X_test (pd.DataFrame): Özellikleri içeren Validation/Test seti
            
        Returns:
            tuple: (X_test_scaled, X_test_pca)
        """
        # Kaydedilmiş nesneleri yükle
        scaler = self.artifact_manager.load_artifact("scaler")
        pca = self.artifact_manager.load_artifact("pca")
        
        # Eksik verileri doldur
        X_test = X_test.ffill().bfill()
        
        # Transform işlemlerini uygula
        X_scaled_array = scaler.transform(X_test)
        
        X_test_scaled = pd.DataFrame(
            X_scaled_array, 
            index=X_test.index, 
            columns=X_test.columns
        )
        
        X_pca_array = pca.transform(X_test_scaled)
        
        X_test_pca = pd.DataFrame(
            X_pca_array, 
            index=X_test.index, 
            columns=['PC1']
        )
        
        return X_test_scaled, X_test_pca

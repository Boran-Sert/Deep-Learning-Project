import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Generator, Tuple

from src.core.config_manager import ConfigurationManager
from src.data.splitter.base import ISplitStrategy


class SkabGroupFoldStrategy(ISplitStrategy):
    """
    SKAB veri seti için GroupKFold stratejisi.
    Veri sızıntısını önlemek için aynı 'source_file' değerine sahip verilerin
    hem train hem test setinde aynı anda bulunmamasını sağlar.
    """

    def __init__(self):
        self.config = ConfigurationManager()
        self.n_splits = self.config.get("experiment.k_folds", 5)

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        GroupKFold uygulayarak train ve test setlerini oluşturur.
        Validation seti ayrıca istenirse, bu train seti içinden daha sonra
        zaman sırası korunarak ayrılabilir.

        Args:
            df (pd.DataFrame): 'source_file' sütununu barındıran DataFrame.

        Yields:
            (train_df, test_df) tuple generator'ı.
        """
        if "source_file" not in df.columns:
            raise ValueError(
                "SkabGroupFoldStrategy gereksinimi olan "
                "'source_file' sütunu bulunamadı."
            )

        gkf = GroupKFold(n_splits=self.n_splits)
        groups = df["source_file"]

        # Scikit-learn indeks (integer) bazlı böler, bu yüzden iloc kullanıyoruz
        for train_idx, test_idx in gkf.split(df, groups=groups):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Zaman sırasını garanti altına almak için index'e göre sıralama
            # (genelde sıralıdır ama tedbiren)
            train_df.sort_index(inplace=True)
            test_df.sort_index(inplace=True)

            yield train_df, test_df

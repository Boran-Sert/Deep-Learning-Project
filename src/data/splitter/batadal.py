import pandas as pd
from typing import Tuple, Generator

from src.data.splitter.base import ISplitStrategy

class BatadalTemporalSplitStrategy(ISplitStrategy):
    """
    BATADAL veri seti için zamansal sıra bozulmadan (shuffle=False) 
    %60 Train, %20 Validation, %20 Test olacak şekilde ardışık (chronological) bölme yapar.
    """

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Zaman sırasını bozmadan DataFrame'i üçe böler.
        
        Args:
            df (pd.DataFrame): Zaman sıralı DataFrame. Index'inin zamana göre 
                               sıralı olduğu varsayılır.
            
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        # Veri setinin indeksine göre sıralı olduğundan emin olalım
        df_sorted = df.sort_index()
        
        n_samples = len(df_sorted)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        yield train_df, val_df, test_df

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setindeki eksik değerleri (NaN) doldurur.
    Zaman serisi projelerinde veri sürekliliğini bozmamak adına varsayılan olarak
    forward-fill (ffill) ve gerekirse backward-fill (bfill) kullanılır.

    Args:
        df (pd.DataFrame): Eksik veri içerebilen DataFrame.

    Returns:
        pd.DataFrame: Eksik değerleri temizlenmiş/doldurulmuş DataFrame.
    """
    # Zaman serisi mantığında en uygun doldurma öncekinin aynısını kopyalamaktır (ffill)
    # Eğer en başta eksik veri varsa bfill ile desteklenir
    df_cleaned = df.ffill().bfill()

    return df_cleaned


def extract_features_and_target(df: pd.DataFrame, target_col: str = "anomaly") -> tuple:
    """
    Veri setini X (özellikler) ve y (hedef değişken) olarak ikiye ayırır.
    Metadata kolonlarını (source_file, source_group vs.) özelliklerden çıkarır.

    Args:
        df (pd.DataFrame): Ham DataFrame
        target_col (str): Hedef değişkenin kolon adı

    Returns:
        tuple: (X_df, y_series)
    """
    # Özellik olarak kullanılmayacak meta veri kolonları (varsa)
    meta_cols = ["source_file", "source_group"]

    # Model girdisi olmayacak tüm kolonları düşür
    cols_to_drop = [target_col] + [c for c in meta_cols if c in df.columns]

    # Eğer target kolon veri setinde hiç yoksa (örn. sadece test verisi) boş dönebilir
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=cols_to_drop)
    else:
        y = pd.Series(0, index=df.index, name=target_col)
        cols_to_drop = [c for c in meta_cols if c in df.columns]
        X = df.drop(columns=cols_to_drop)

    return X, y

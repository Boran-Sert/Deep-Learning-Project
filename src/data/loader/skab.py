import os
import glob
import pandas as pd
from typing import List

from src.core.config_manager import ConfigurationManager
from src.data.loader.base import IDataLoader


class SkabLoader(IDataLoader):
    """
    SKAB (Skoltech Anomaly Benchmark) veri setini yükleyen sınıf.
    valve1 ve valve2 dizinlerindeki csv dosyalarını birleştirir,
    metadata (source_group, source_file) ekler ve indeksi datetime yapar.
    """

    def __init__(self):
        self.config = ConfigurationManager()

    def load_data(self) -> pd.DataFrame:
        # Proje kök dizinini bul
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        valve1_path = os.path.join(project_root, self.config.get("paths.skab_valve1"))
        valve2_path = os.path.join(project_root, self.config.get("paths.skab_valve2"))

        dataframes: List[pd.DataFrame] = []

        # Valve 1 Okuma
        self._read_valve_directory(valve1_path, "valve1", dataframes)
        # Valve 2 Okuma
        self._read_valve_directory(valve2_path, "valve2", dataframes)

        if not dataframes:
            raise FileNotFoundError(
                "SKAB veri dosyaları bulunamadı. Lütfen dizin yollarını kontrol edin."
            )

        # Tüm dataframe'leri birleştir
        df_combined = pd.concat(dataframes, ignore_index=False)

        # Sütun isimlerini standartlaştır
        if "anomaly" not in df_combined.columns:
            if "changepoint" in df_combined.columns:
                df_combined.rename(columns={"changepoint": "anomaly"}, inplace=True)
            else:
                raise ValueError("SKAB veri setinde 'anomaly' kolonu bulunamadı!")

        # İndeksin adını temizle
        df_combined.index.name = "datetime"

        # Zamana göre sırala (her dosyanın kendi içinde zaten sıralı olduğu
        # varsayılır, genel sıralama yapılabilir)
        # Ancak farklı valflerde çakışan zamanlar olabilir, source bazında
        # sıralamak daha iyi olabilir.
        df_combined.sort_index(inplace=True)

        return df_combined

    def _read_valve_directory(
        self, dir_path: str, group_name: str, df_list: List[pd.DataFrame]
    ) -> None:
        if not os.path.exists(dir_path):
            return

        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        for file_path in csv_files:
            file_name = os.path.basename(file_path)

            # SKAB dosyaları genellikle index='datetime' ve ayrıştırıcı
            # olarak sep=';' ile gelir.
            # Virgülle ayrılmış CSV is default, deniyoruz.
            try:
                df = pd.read_csv(
                    file_path, index_col="datetime", parse_dates=True, sep=";"
                )
            except ValueError:
                # Eger ; degilse ve datetime index bulunamazsa default ile dene
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Metadata ekle
            df["source_group"] = group_name
            df["source_file"] = file_name

            df_list.append(df)

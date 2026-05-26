import os
import pandas as pd
from src.core.config_manager import ConfigurationManager
from src.data.loader.base import IDataLoader


class BatadalLoader(IDataLoader):
    """
    BATADAL veri setini (sadece Training Dataset 2) yükleyen sınıf.

    Zaman sütununu indeks olarak ayarlar ve hedef sütununu
    'anomaly' olarak standartlaştırır.
    """

    def __init__(self):
        self.config = ConfigurationManager()

    def load_data(self) -> pd.DataFrame:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        batadal_path = os.path.join(
            project_root, self.config.get("paths.batadal_dataset")
        )

        if not os.path.exists(batadal_path):
            raise FileNotFoundError(f"BATADAL veri dosyası bulunamadı: {batadal_path}")

        # Sadece Training Dataset 2'yi okuduğumuzdan emin olalım
        if "BATADAL_dataset04.csv" not in os.path.basename(batadal_path):
            raise ValueError(
                "Sadece Training Dataset 2 (BATADAL_dataset04.csv) "
                f"kullanılmalıdır! Verilen yol: {batadal_path}"
            )

        # BATADAL csv okuma
        df = pd.read_csv(batadal_path)

        # Sütun isimlerindeki boşlukları temizleyelim
        df.columns = df.columns.str.strip()

        # Datetime sütununu index yapalım
        if "DATETIME" in df.columns:
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], dayfirst=True)
            df.set_index("DATETIME", inplace=True)
            df.index.name = "datetime"
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

        # Hedef kolonunu 'anomaly' olarak standartlaştır
        if "ATT_FLAG" in df.columns:
            df.rename(columns={"ATT_FLAG": "anomaly"}, inplace=True)

        # Eğer 'anomaly' kolonu yoksa hata fırlat
        if "anomaly" not in df.columns:
            raise ValueError(
                "BATADAL veri setinde hedef (ATT_FLAG -> anomaly) kolonu bulunamadı!"
            )

        # 🚨 KESİN ETİKET TEMİZLEME KANCASI
        # 0, 0.0, NORMAL dışındaki tüm siber saldırı, anomali veya bozuk etiketleri tam olarak 1'e indirgiyoruz
        df["anomaly"] = df["anomaly"].apply(lambda x: 1 if str(x).strip() not in ["0", "0.0", "NORMAL", "Normal"] else 0)

        df.sort_index(inplace=True)
        return df
from abc import ABC, abstractmethod
import pandas as pd


class IDataLoader(ABC):
    """
    Tüm veri setleri için veri yükleme arayüzü.
    Factory Pattern ile üretilen nesneler bu arayüzü uygulayacaktır.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Veri setini okuyup pandas DataFrame olarak döndürür.
        """
        pass

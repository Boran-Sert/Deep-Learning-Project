from src.data.loader.base import IDataLoader
from src.data.loader.batadal import BatadalLoader
from src.data.loader.skab import SkabLoader


class DataLoaderFactory:
    """
    Veri yükleyicileri (Loaders) üretmek için Factory sınıfı.
    """

    @staticmethod
    def get_loader(dataset_name: str) -> IDataLoader:
        """
        İstenen veri setinin ismine göre ilgili Loader sınıfını döndürür.

        Args:
            dataset_name: "skab" veya "batadal"

        Returns:
            IDataLoader nesnesi
        """
        dataset_name = dataset_name.lower().strip()

        if dataset_name == "skab":
            return SkabLoader()
        elif dataset_name == "batadal":
            return BatadalLoader()
        else:
            raise ValueError(
                f"Bilinmeyen veri seti türü: {dataset_name}. "
                "Lütfen 'skab' veya 'batadal' kullanın."
            )

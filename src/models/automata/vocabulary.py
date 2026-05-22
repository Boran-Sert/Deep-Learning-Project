from typing import List, Dict
from src.core.artifact_manager import ExperimentArtifactManager


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    İki string arasındaki Levenshtein (düzenleme) mesafesini hesaplar.
    Herhangi bir harici kütüphane bağımlılığı olmaması için saf Python ile yazılmıştır.
    Pencere boyutları (kelime uzunluğu) genelde 5-10 karakter olduğu için
    performans sorunu yaratmaz.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class UnseenHandler:
    """
    Test aşamasında karşılaşılan ancak eğitim aşamasındaki sözlükte (vocabulary)
    bulunmayan kelimeleri en yakın sözlük elemanına haritalayan sınıf.
    """

    @staticmethod
    def handle_unseen(word: str, vocabulary: Dict[str, int]) -> str:
        """
        Verilen kelimeye Levenshtein mesafesi en düşük olan sözlük elemanını döner.

        Args:
            word (str): Sözlükte bulunmayan (unseen) kelime
            vocabulary (Dict[str, int]): Eğitim aşamasında oluşturulan sözlük

        Returns:
            str: En yakın kelime
        """
        if not vocabulary:
            raise ValueError("Sözlük (vocabulary) boş olamaz!")

        best_match: str = ""
        min_distance = float("inf")

        for vocab_word in vocabulary.keys():
            dist = levenshtein_distance(word, vocab_word)
            if dist < min_distance:
                min_distance = dist
                best_match = vocab_word
                # Eğer tam eşleşme (uzaklık 0) bulunursa direkt çık
                # (genelde imkansız ama tedbir)
                if min_distance == 0:
                    break

        return best_match


class VocabularyManager:
    """
    Eğitim verisinden çıkarılan tüm SAX kelimelerini (pattern) ve frekanslarını
    yöneten sınıftır. ExperimentArtifactManager ile diske kaydedilir/okunur.
    """

    def __init__(self, experiment_id: str = "default"):
        self.experiment_id = experiment_id
        self.artifact_manager = ExperimentArtifactManager(experiment_id=experiment_id)
        self.vocabulary: Dict[str, int] = {}

    def build_vocabulary(self, words: List[str]) -> None:
        """
        Verilen kelime listesinden frekans sözlüğü oluşturur.
        """
        self.vocabulary = {}
        for word in words:
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

    def get_state(self, word: str) -> str:
        """
        Kelimenin sözlükteki halini döndürür. Yoksa UnseenHandler devreye girer.
        """
        if word in self.vocabulary:
            return word
        else:
            return UnseenHandler.handle_unseen(word, self.vocabulary)

    def save(self, artifact_name: str = "automata_vocab"):
        """Sözlüğü JSON olarak diske kaydeder."""
        self.artifact_manager.save_dict_artifact(self.vocabulary, artifact_name)

    def load(self, artifact_name: str = "automata_vocab"):
        """Diskteki JSON sözlüğü belleğe yükler."""
        self.vocabulary = self.artifact_manager.load_dict_artifact(artifact_name)

from typing import List, Union

from .classy_spacy import (
    ClassyExternal,
    ClassySkeletonFewShot,
    ClassySkeletonFewShotMultiLabel,
)


class ClassyStandalone(ClassyExternal):
    def __init__(
        self,
        model: str,
        device: str,
        data: dict,
        config: Union[dict, None] = None,
    ):
        self.data = data
        self.model = model
        self.device = device
        self.set_config(config)
        self.set_embedding_model()
        self.set_training_data()
        self.set_classification_model()

    def __call__(self, text: str) -> dict:
        """predict the class for an input text

        Args:
            text (str): an input text

        Returns:
            dict: a key-class proba-value dict
        """
        embeddings = self.get_embeddings([text])

        return self.get_prediction(embeddings)[0]

    def pipe(self, text: List[str]) -> List[dict]:
        """retrieve predictions for multiple texts

        Args:
            text (List[str]): a list of texts

        Returns:
            List[dict]: list of key-class proba-value dict
        """
        embeddings = self.get_embeddings(text)

        return self.get_prediction(embeddings)


class ClassySentenceTransformerFewShot(ClassyStandalone, ClassySkeletonFewShot):
    pass


class ClassySentenceTransformerMultiLabel(ClassyStandalone, ClassySkeletonFewShotMultiLabel):
    pass


def classySentenceTransformer(
    data: dict,
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: str = None,
    config: Union[dict, None] = None,
    multi_label: bool = False,
):
    if multi_label:
        return ClassySentenceTransformerMultiLabel(model, device, data, config)
    else:
        return ClassySentenceTransformerFewShot(model, device, data, config)

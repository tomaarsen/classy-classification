import logging
from typing import Union

from spacy.language import Language

from .classifiers.classy_spacy import (
    ClassySpacyExternalFewShot,
    ClassySpacyExternalFewShotMultiLabel,
    ClassySpacyExternalZeroShot,
    ClassySpacyInternalFewShot,
    ClassySpacyInternalFewShotMultiLabel,
)
from .classifiers.classy_standalone import classySentenceTransformer as ClassyClassifier

__all__ = [
    "ClassyClassifier",
    "ClassySpacyExternalFewShot",
    "ClassySpacyExternalFewShotMultiLabel",
    "ClassySpacyExternalZeroShot",
    "ClassySpacyInternalFewShot",
    "ClassySpacyInternalFewShotMultiLabel",
]

logging.captureWarnings(True)


@Language.factory(
    "text_categorizer",
    default_config={
        "data": None,
        "model": None,
        "device": "cpu",
        "config": None,
        "cat_type": "few",
        "multi_label": False,
        "include_doc": True,
        "include_sent": False,
    },
)
def make_text_categorizer(
    nlp: Language,
    name: str,
    data: Union[dict, list],
    device: str,
    config: dict = None,
    model: str = None,
    cat_type: str = "few",
    multi_label: bool = False,
    include_doc: bool = True,
    include_sent: bool = False,
):
    if model == "spacy":
        if cat_type == "zero":
            raise NotImplementedError("Cannot use spacy internal embeddings with zero-shot classification")
        if multi_label:
            return ClassySpacyInternalFewShotMultiLabel(
                nlp=nlp,
                name=name,
                data=data,
                config=config,
                include_doc=include_doc,
                include_sent=include_sent,
            )
        else:
            return ClassySpacyInternalFewShot(
                nlp=nlp,
                name=name,
                data=data,
                config=config,
                include_doc=include_doc,
                include_sent=include_sent,
            )
    else:
        if cat_type == "zero":
            if model:
                return ClassySpacyExternalZeroShot(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    model=model,
                    include_doc=include_doc,
                    include_sent=include_sent,
                    multi_label=multi_label,
                )
            else:
                return ClassySpacyExternalZeroShot(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    include_doc=include_doc,
                    include_sent=include_sent,
                    multi_label=multi_label,
                )
        if multi_label:
            if model:
                return ClassySpacyExternalFewShotMultiLabel(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    model=model,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
            else:
                return ClassySpacyExternalFewShotMultiLabel(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
        else:
            if model:
                return ClassySpacyExternalFewShot(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    model=model,
                    config=config,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )
            else:
                return ClassySpacyExternalFewShot(
                    nlp=nlp,
                    name=name,
                    data=data,
                    device=device,
                    config=config,
                    include_doc=include_doc,
                    include_sent=include_sent,
                )

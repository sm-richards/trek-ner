from typing import Mapping, Sequence, Dict, Optional

from spacy.language import Language
from spacy.tokens import Doc

from hw3utils import PRF1


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    raise NotImplementedError


def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    raise NotImplementedError

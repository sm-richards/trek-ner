from typing import Mapping, Sequence, Dict, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from hw3utils import PRF1
from collections import defaultdict
from typing import Sequence, Dict, Optional, List
import spacy
from spacy.tokens import Doc, Span, Token
from hw3utils import FeatureExtractor, ScoringCounts, ScoringEntity, EntityEncoder, PUNC_REPEAT_RE, PRF1
from pymagnitude import *
from collections import defaultdict, Counter
from cosi217.ingest import *
import regex
import pycrfsuite
from cosi217.debug import print_ents
from decimal import ROUND_HALF_UP, Context
import json
from nltk.stem.porter import *
import random
import jsonlines
from gensim.models import Word2Vec

stemmer = PorterStemmer()


class BiasFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if relative_idx == 0:
            features["bias"] = 1.0


class TokenFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        features[f"tok[{str(relative_idx)}]={token}"] = 1.0


class UppercaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if token.isupper():
            features[f"uppercase[{str(relative_idx)}]"] = 1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if token.istitle():
            features[f"titlecase[{str(relative_idx)}]"] = 1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if token.istitle() and ((current_idx + relative_idx) == 0):
            features[f"initialtitlecase[{str(relative_idx)}]"] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if regex.match(PUNC_REPEAT_RE, token):
            features[f"punc[{str(relative_idx)}]"] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if re.search(r"\d", token):
            features[f"digit[{str(relative_idx)}]"] = 1.0


class SuffixFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if len(token) > 4:
            return token[-3:]


class PrefixFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        if len(token) > 4:
            return token[:2]


class StemFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        return stemmer.stem(token)


class WordShapeFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str],
                features: Dict[str, float]):
        shape = []
        for letter in list(token):
            if letter.isupper():
                shape.append("X")
            elif letter.islower():
                shape.append("x")
            elif letter.isnumeric():
                shape.append("0")
            else:
                shape.append(letter)
        features[f"shape[{str(relative_idx)}]={''.join(shape)}"] = 1.0


class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.vectors = Word2Vec.load(vectors_path)
        self.scale = scaling

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            try:
                token = token.strip().lower()
                vector = self.vectors[token]
                ftr_keys = ["v" + str(i) for i in range(len(vector))]
                values = list(vector * self.scale)
                features.update(zip(ftr_keys, values))
            except KeyError:
                pass


class WordVectorFeature2(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0, continuous: bool = True) -> None:
        self.vectors = Word2Vec.load(vectors_path)
        self.scale = scaling
        self.continuous = continuous

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            token = token.strip().lower()
            try:
                vector_sims = self.vectors.wv.similar_by_word(token)
                if self.continuous:
                    ftrs = [(f"similarity to {str(token)}=", value * self.scale) for token, value in vector_sims]
                else:
                    ftrs = [(f"similarity to {str(token)}=", 1.0) for token, value in vector_sims]
                features.update(ftrs)
            except KeyError:
                pass


class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size

    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        feature_dicts = []
        for curr_idx, token in enumerate(tokens):
            feature_dict = {}
            start = curr_idx - self.window_size if (curr_idx - self.window_size) >= 0 else 0
            end = curr_idx + self.window_size + 1 if (curr_idx + self.window_size) + 1 <= len(tokens) else len(tokens)
            token_window = tokens[start: end]
            rel_idx = -1*self.window_size + ((2*self.window_size + 1) - len(token_window)) \
                if (curr_idx + self.window_size) <= len(tokens)-1 else -1*self.window_size
            for tok in token_window:
                for extractor in self.feature_extractors:
                    extractor.extract(tok, curr_idx, rel_idx, tokens, feature_dict)
                rel_idx += 1
            feature_dicts.append(feature_dict)
        return feature_dicts


class BILOUEncoder(EntityEncoder):
    def encode(self, tokens):
        ents = [token.ent_type_ for token in tokens]
        bilou_tags = []

        prev_type = None
        for i, type in enumerate(ents):
            next_type = ents[i+1] if i < len(ents) - 1 else None
            if type == '':  # O
                bilou_tags.append('O')
            else:  # B, I, L, or U
                if prev_type == '' or prev_type != type:  # B or U
                    if next_type == '' or next_type != type:  # U
                        bilou_tags.append('U-')
                    else:  # B
                        bilou_tags.append('B-')
                else:  # I or L
                    if next_type == '' or next_type != type:  # L
                        bilou_tags.append('L-')
                    else:  # I
                        bilou_tags.append('I-')
            prev_type = type

        encoding = [bilou_tags[i] + token.ent_type_ for i, token in enumerate(tokens)]
        return encoding


class BIOEncoder(EntityEncoder):
    def encode(self, tokens):
        return ['O' if token.ent_type_ is '' else token.ent_iob_ + '-' + token.ent_type_ for token in tokens]

    def encode_cautious(self, tokens):
        ents = [token.ent_type_ for token in tokens]
        bio_tags = []

        prevv = None
        for i, item in enumerate(ents):
            if item == '':  # O
                bio_tags.append('O')
            else:  # B or I
                if prevv == '' or prevv != item:  # B
                    bio_tags.append('B-')
                else:  # I
                    bio_tags.append('I-')
            prevv = item

        return [bio_tags[i] + token.ent_type_ for i, token in enumerate(tokens)]


class IOEncoder(EntityEncoder):
    def encode(self, tokens):
        return ['I-' + token.ent_type_ if token.ent_type_ is not '' else 'O' for token in tokens]


def decode(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    spandict = defaultdict(list)
    i = 0
    prev_type = 'O'
    for index, label in enumerate(labels):
        tag = label[0]
        enttype = label[2:]
        doc_index = tokens[index].i
        if label != 'O':
            if tag == 'B' or prev_type != enttype:
                i = index
            spandict[(enttype, i)].append(doc_index)
        elif label == 'O':
            i = index + 1
        prev_type = enttype
    spanlist = []
    for label, doc_span in spandict.items():
        spanlist.append(Span(doc, doc_span[0], doc_span[-1] + 1, label[0]))

    return spanlist


class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self.entity_encoder = encoder
        self.tagger = pycrfsuite.Tagger()

    @property
    def encoder(self) -> EntityEncoder:
        return self.entity_encoder

    def train(self, docs, algorithm: str, params: dict, path: str) -> None:
        trainer = pycrfsuite.Trainer(algorithm, verbose=False)
        trainer.set_params(params)
        for doc in docs:
            for sent in doc.sents:
                token_strings = [str(token) for token in list(sent)]
                features = self.feature_extractor.extract(token_strings)
                tags = self.entity_encoder.encode(list(sent))
                trainer.append(features, tags)
        trainer.train(path)
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        entities = []
        for sent in doc.sents:
            labels = self.predict_labels([str(token) for token in list(sent)])
            spans = decode(labels, list(sent), doc)
            entities.extend(spans)
        doc.ents = entities
        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        features = self.feature_extractor.extract(tokens)
        return self.tagger.tag(features)


def span_prf1(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> Dict[str, PRF1]:

    counts = defaultdict(lambda: defaultdict(int))
    for test_doc, ref_doc in zip(test_docs, reference_docs):

        test_spans = [(ent.start, ent.end, ent.label_) if typed else (ent.start, ent.end, "") for ent in test_doc.ents]
        ref_spans = [(ent.start, ent.end, ent.label_) if typed else (ent.start, ent.end, "") for ent in ref_doc.ents]

        for span in test_spans:
            if span in ref_spans:           # true positive
                counts[span[2]]['tp'] += 1
                counts["all"]['tp'] += 1
            else:                           # false positive
                counts[span[2]]['fp'] += 1
                counts["all"]['fp'] += 1
        for span in ref_spans:
            if span not in test_spans:      # false negative
                counts[span[2]]['fn'] += 1
                counts["all"]['fn'] += 1

    prf1_dict = {}
    for ent_type, count in counts.items():
        p = count['tp'] / (count['tp'] + count['fp']) if (count['tp'] + count['fp']) != 0 else 0.0
        r = count['tp'] / (count['tp'] + count['fn']) if (count['tp'] + count['fn']) != 0 else 0.0
        f1 = (2 * (p * r) / (p + r)) if (p + r) != 0 else 0.0
        prf1 = PRF1(p, r, f1)
        prf1_dict[ent_type] = prf1

    if typed:
        prf1_dict[""] = prf1_dict.pop("all")
    else:
        del prf1_dict["all"]

    return prf1_dict


def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:

    tp, fp, fn = [], [], []

    for test_doc, ref_doc in zip(test_docs, reference_docs):
        test_spans = [(ent.start, ent.end, ent.label_) if typed
                      else (ent.start, ent.end, "") for ent in test_doc.ents]
        ref_spans = [(ent.start, ent.end, ent.label_) if typed
                     else (ent.start, ent.end, "") for ent in ref_doc.ents]

        for span in test_spans:
            if span in ref_spans:  # true positive
                token_tu = tuple(str(test_doc[span[0]:span[1]]).split())
                tp.append(ScoringEntity(token_tu, span[2]))
            else:  # false positive
                token_tu = tuple(str(test_doc[span[0]:span[1]]).split())
                fp.append(ScoringEntity(token_tu, span[2]))
        for span in ref_spans:
            if span not in test_spans:  # false negative
                token_tu = tuple(str(ref_doc[span[0]:span[1]]).split())
                fn.append(ScoringEntity(token_tu, span[2]))

    return ScoringCounts(Counter(tp), Counter(fp), Counter(fn))


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:

    text = doc_json['text']
    text = text.replace(")[", ") ")
    text = text.replace("\n", " ")
    text = text.replace(".)", ". ")

    doc = nlp(text)
    tokens = list(doc.__iter__())

    ents = []
    if doc_json['labels'] is [] and doc_json['annotation_approver'] is None:
        raise ValueError
    else:
        for [char_start, char_end, label] in doc_json['labels']:
            token_indices = []
            for char in range(char_start, char_end):
                for token in tokens:
                    if token.idx <= char < token.idx + len(token):
                        token_indices.append(token.i)
                    elif token.idx - 1 == char and doc_json['text'][token.idx - 1] == ' ':
                        token_indices.append(token.i)
            if token_indices is []:
                raise ValueError
            span = Span(doc, token_indices[0], token_indices[-1] + 1, label)
            ents.append(span)
    doc.ents = ents
    return doc


def span_prf1_type_map(reference_docs: Sequence[Doc], test_docs: Sequence[Doc],
                       type_map: Optional[Mapping[str, str]]=None) -> Dict[str, PRF1]:

    mapped_ents = type_map.keys() if type_map is not None else {}
    counts = defaultdict(lambda: defaultdict(int))
    for test_doc, ref_doc in zip(test_docs, reference_docs):

        test_spans = [(ent.start, ent.end, type_map[ent.label_] if ent.label_ in mapped_ents else ent.label_)
                      for ent in test_doc.ents]
        ref_spans = [(ent.start, ent.end, type_map[ent.label_] if ent.label_ in mapped_ents else ent.label_)
                     for ent in ref_doc.ents]

        for span in test_spans:
            if span in ref_spans:  # true positive
                counts[span[2]]['tp'] += 1
                counts[""]['tp'] += 1
            else:  # false positive
                counts[span[2]]['fp'] += 1
                counts[""]['fp'] += 1
        for span in ref_spans:
            if span not in test_spans:  # false negative
                counts[span[2]]['fn'] += 1
                counts[""]['fn'] += 1

    prf1_dict = {}
    for ent_type, count in counts.items():
        p = count['tp'] / (count['tp'] + count['fp']) if (count['tp'] + count['fp']) != 0 else 0.0
        r = count['tp'] / (count['tp'] + count['fn']) if (count['tp'] + count['fn']) != 0 else 0.0
        f1 = (2 * (p * r) / (p + r)) if (p + r) != 0 else 0.0
        prf1 = PRF1(p, r, f1)
        prf1_dict[ent_type] = prf1

    return prf1_dict


def read_in_train_dev_data(file, nlp):
    docs = []
    with open(file, 'r') as datafile:
        for line in datafile.readlines():
            annotation = json.loads(line)
            docs.append(ingest_json_document(annotation, nlp))
    return docs[:565], docs[565:]  # train data, dev data


def read_in_test_data(file, nlp):
    docs = []
    with open(file, 'r') as datafile:
        for line in datafile.readlines():
            annotation = json.loads(line)
            docs.append(ingest_json_document(annotation, nlp))
    return docs  # train data, dev data


# MAIN -----------------------------------------------------------------------------------------------------------------
def main() -> None:
    train_dev_data = "annotated_data.jsonl"
    test_data = "fics_annotated_all.jsonl"
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    train, dev = read_in_train_dev_data(train_dev_data, nlp)
    test = read_in_test_data(test_data, nlp)

    # Delete annotation on test
    for doc in test:
        doc.ents = []

    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            [
                BiasFeature(),
                TokenFeature(),
                UppercaseFeature(),
                TitlecaseFeature(),
                InitialTitlecaseFeature(),
                PunctuationFeature(),
                DigitFeature(),
                WordShapeFeature(),
                WordVectorFeature('trek_w2v.model', scaling=1.0)
            ],
            1,
        ),
        BILOUEncoder(),
    )
    crf.train(train, "ap", {"max_iterations": 40}, "tmp.model")
    test = [crf(doc) for doc in test]

    # Load valid again to eval
    test_gold = read_in_test_data(test_data, nlp)
    print("Type\tPrec\tRec\tF1", file=sys.stderr)
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    # Set typed=False for untyped scores
    scores = span_prf1(test_gold, test, typed=True)
    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"

        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)

    counts = span_scoring_counts(test_gold, test, typed=True)
    print("True positives ", sum(counts.true_positives.values()))
    print("False positives ", sum(counts.false_positives.values()))
    print("False negatives ", sum(counts.false_negatives.values()))


if __name__ == "__main__":
    main()



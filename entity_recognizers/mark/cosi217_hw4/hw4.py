from typing import Mapping, Sequence, Dict, Optional

from spacy.language import Language
from spacy.tokens import Doc

# from hw3utils import PRF1
from hw3utils import PRF1, FeatureExtractor, ScoringCounts, ScoringEntity


#HW3 imports:
from pymagnitude import Magnitude
from collections import defaultdict

from typing import List, Dict, Counter, Optional, TextIO, Generator
import spacy
from spacy.gold import spans_from_biluo_tags, iob_to_biluo

from spacy.language import Language
from itertools import chain

#HW2 imports:
from abc import ABC, abstractmethod
from typing import Sequence, Dict, List, NamedTuple

import re
from typing import Iterable, Sequence, Tuple, List, Dict

from nltk import ConfusionMatrix
from regex import regex
from spacy.tokens import Span, Doc, Token
# import hw2utils
# from hw2utils import FeatureExtractor, EntityEncoder, PRF1
import pycrfsuite

import json
# import time





#Create a document from the json file. Requires mapping character spans to token spans
def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    # tempo = time.clock()
    labels = doc_json["labels"]
    if len(labels) == 0 and not doc_json["annotation_approver"]:
        raise ValueError
    for label in labels:
        # specs say < 0 doesn't happen, so this is the only value error I see.
        if label[0] > len(doc_json["text"]) or label[1] > len(doc_json["text"]):
            raise ValueError
        # LOL whitespace too I guess
        elif doc_json["text"][label[0]:label[1]] == " " * (label[1] - label[0]):
            print(label, doc_json["text"][label[0]:label[1]])
            raise  ValueError
    to_return = nlp(doc_json["text"])


    entities = []
    last_label = [-1]
    for sent in to_return.sents:
        toks = list(sent)
        tags = ["O"] * len(toks)
        for t in range(len(toks)):
            tok = toks[t]
            #print(tok.idx, tok.idx + len(tok))
            for label in labels: # I feel kinda dirty for the double nesting, but neither one is very long.
                if (label[0] <= tok.idx < label[1] or label[0] <= tok.idx + len(tok) < label[1]):
                    if last_label[0] == label[0]: # Slightly faster than comparing the whole list
                        tags[t] = "I-" + label[2]
                    else:
                        tags[t] = "B-" + label[2] # BIO is sufficient for translator.
                        last_label = label # Labels are linear, so weird edge cases shouldn't come up

        #print(tags, "\n", toks, "\n", labels)
        to_append = decode_bilou(tags, sent, to_return)
        entities.extend(to_append)
    to_return.ents = entities
    #print(time.clock() - tempo)
    return to_return


def remap(ents, mapping, count):

    #LOL one line now
    # Tuple in order to differentiate docs quicker later. Messes with readability though.
    return tuple([(Span(ent.doc, ent.start, ent.end, label = mapping[ent.label_]), count)
                  if ent.label_ in mapping else (ent, count) for ent in ents]) #(ent, count)

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:

    # Make good use of sets here.
    # print(reference_docs[0].ents)
    # print(test_docs[0].ents)
    #type_map = {"GPE": "GPE_LOC", "LOC": "GPE_LOC"}

    if len(reference_docs) != len(test_docs):
        # Should always be same length
        raise ValueError
    refset = set()
    tesset = set()

    label_options = set([""])
    if type_map == None:
        type_map = {}

    for i in range(len(reference_docs)):
        ref = reference_docs[i]
        tes = test_docs[i]
        refents = remap(ref.ents, type_map, i) # i to keep docs discernable.
        tesents = remap(tes.ents, type_map, i)

        for ent in refents:
            refset.add(ent)
            label_options.add(ent[0].label_)
        for ent in tesents:
            tesset.add(ent)
            label_options.add(ent[0].label_) # eh, why not

    to_return = {}
    for label in label_options:
        if label == "":
            refset1 = set([(x[0].start, x[0].end, x[0].label_, x[1]) for x in refset])
            tesset1 = set([(x[0].start, x[0].end, x[0].label_, x[1]) for x in tesset])
        else:
            refset1 = set([(x[0].start, x[0].end, x[0].label_, x[1]) for x in refset if x[0].label_ == label])
            tesset1 = set([(x[0].start, x[0].end, x[0].label_, x[1]) for x in tesset if x[0].label_ == label])
        #print([x for x in refset1], [x for x in tesset1])

        # I wanna use sets, but it's not playing nice with checking the label :(
        #set([x for x in refset1 for y in tesset1 if x == y and x.label_ == y.label_])#
        tp = refset1.intersection(tesset1) #['EPG', 'COL', 'REP']
        fp = tesset1 - tp
        fn = refset1 - tp

        # print(label + "TP:", tp, tp)
        # print(label + "FP:", fp)
        # print(label + "FN:", fn)

        if len(tp) == 0:
            p = 0
            r = 0
        else:
            p = (float(len(tp)) / (len(tp) + len(fp)))
            r = (float(len(tp)) / (len(tp) + len(fn)))
        if p + r == 0:
            f1 = 0
        else:
            f1 = (2 * p * r) / (p + r)
        #f1 = 2 / (1.0 / p + 1.0 / r)
        to_return[label] = PRF1(precision= p, recall=r, f1=f1)

    #print(tp, fp, fn)
    #Dict[str, PRF1]
    # print(to_return)
    return to_return

    #raise NotImplementedError





#HW3 stuff below
class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:

        self.mag = Magnitude(vectors_path, normalized=False)
        self.scaling = scaling



    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        # let's just write the suggested lines right here, and see how much they hook up
        if relative_idx == 0:
            values = self.mag.query(token) * self.scaling # Scaling matters lol
            #print(token, current_idx, tokens, values)
            keys = ["v" + str(i) for i in range(len(values))]
            features.update(zip(keys, values))


class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        if (not (use_full_paths or use_prefixes)):
            raise ValueError
        if use_full_paths: self.path = "use_full_paths"
        else: self.path = "use_prefixes"
        self.prefixes = prefixes
        # load cluster path
        # cluster, word, frequency. Only care about cluster/word pairs
        # Make a dict, word:cluster
        # ... Could make binary to save space, but meh
        self.cluster_dict = defaultdict(str)
        with open(clusters_path) as file:
            #data = file.read()
            for line in file:
                linesplit = line.split("\t")
                #print(linesplit)
                if len(linesplit) > 1: #empty line nonsense
                    self.cluster_dict[linesplit[1]] = linesplit[0]
        #print(self.cluster_dict)

    #
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:

            path = self.cluster_dict[token]
            if path == "": # Handle OOV words
                path = "*UNK*"
            if self.path == "use_full_paths":
                cpath = "cpath=" + path
                features[cpath] = 1.0
            else:
                if self.prefixes != None: prefixes = [x for x in self.prefixes if x < len(path)]
                else: prefixes = range(1,len(path) + 1) # silly off by 1s
                for pre in prefixes:
                    cprefix = "cprefix" + str(pre) + "=" +  path[:pre]
                    features[cprefix] = 1.0


def doc_to_spans(doc: Sequence[Doc]) -> List:
    # Return a list of tuples. (start, stop, tag)
    to_return = []
    cur_start = 0
    cur = 0
    last = "O"
    for token in range(len(doc)):
        if doc[token] == "O":
            # end current span
            if last != "O" and cur != cur_start:
                to_return.append((cur_start, cur, last[2:]))
            cur_start = token
            last = doc[token]
        else:
            if last == "O":
                cur_start = token
            elif last[2:] != doc[token][2:]:
                # New entity AND close old
                to_return.append((cur_start, cur, last[2:]))
                cur_start = token
            else:
                #Same token
                pass
        cur = token

    return to_return



def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:

    # true_positives: Counter[ScoringEntity]
    # false_positives: Counter[ScoringEntity]
    # false_negatives: Counter[ScoringEntity]

    # docs = [[["George", "Washington"]], [["Maryland"]], [["American"]]]
    # correct_bio = [["B-PER", "I-PER"], ["B-GPE"], ["B-MISC"]]
    # incorrect_bio1 = [["B-PER", "O"], ["B-GPE"], ["B-MISC"]]
    # incorrect_bio2 = [["B-PER", "I-PER"], ["B-GPE"], ["B-GPE"]]

    # class ScoringEntity(NamedTuple):
    #     tokens: Tuple[str, ...]
    #     entity_type: str

    perfect_list = [] # Lists of ScoringEntity
    false_negatives = []
    false_positives = []

    for doc in range(len(reference_docs)):
        if(len(reference_docs[doc]) != len(test_docs[doc]) ):
            print("Documents of unequal length!")
            continue # idk how best to handle
        if not typed:
            ents1 = set([(tuple(x.text.split()), "") for x in reference_docs[doc].ents])
            ents2 = set([(tuple(x.text.split()), "") for x in test_docs[doc].ents])
        else:
            ents1 = set([(tuple(x.text.split()), x.label_) for x in reference_docs[doc].ents])
            ents2 = set([(tuple(x.text.split()), x.label_) for x in test_docs[doc].ents])

        # Figured I'd try sets this time. Likely slow.
        perfect_list.extend(list(ents1 & ents2))
        false_negatives.extend(list((ents1 | ents2) - ents2 ))
        false_positives.extend(list((ents1 | ents2) - ents1 ))

    perfect_list1 = [ScoringEntity(x[0], x[1]) for x in perfect_list]
    false_negatives1 = [ScoringEntity(x[0], x[1]) for x in false_negatives]
    false_positives1 = [ScoringEntity(x[0], x[1]) for x in false_positives]

    to_return = ScoringCounts(Counter(perfect_list1), Counter(false_positives1), Counter(false_negatives1))
    print("\n\nRETURNED", to_return.true_positives, "\n", to_return.false_negatives, "\n", to_return.false_positives, "\n\n")
    return to_return


def performance_analysis_comments1() -> None:
    """
    TODO: Describe your findings here

My thing still has the problem where it does 100 iterations instead of 40. Numbers may be marginally higher or less sensitive to feature changes as a result.

BASELINE:
SCORES:
precision 0.8843109869646183
recall 0.894724446537918
f1 0.8894872395223602
44.035401 Seconds taken

-Bias
SCORES:
precision 0.882544103992572
recall 0.8954309938765898
f1 0.888940846387655
39.934328 Seconds taken
~Ha! useless

-TokenFeature
SCORES:
precision 0.6252414681262073
recall 0.686057465850212
f1 0.6542391914654688
41.400385 Seconds taken
~Very important

-Upper
SCORES:
precision 0.8836287313432836
recall 0.8923692887423458
f1 0.8879775017576752
41.381531 Seconds taken
~Unhelpful

-Title
SCORES:
precision 0.8835087719298246
recall 0.889543099387659
f1 0.8865156671752141
40.995779 Seconds taken

-InitTit
SCORES:
precision 0.8798233379823338
recall 0.8914272256241168
f1 0.8855872718764622
40.173582999999994 Seconds taken

-Punk
SCORES:
precision 0.8888369920597852
recall 0.8963730569948186
f1 0.8925891181988744
39.489774 Seconds taken
~Literally improved

-Digit
SCORES:
precision 0.8839431367979492
recall 0.8933113518605746
f1 0.8886025535902541
41.319908 Seconds taken

-Shape
SCORES:
precision 0.8697290930506478
recall 0.8695242581252944
f1 0.8696266635260864
21.001756 Seconds taken
~Noticeably worse


-MINUS SHAPE BEGINS

-Bias
SCORES:
precision 0.8686251468860164
recall 0.8704663212435233
f1 0.8695447594400659
19.462709 Seconds taken

-Token
SCORES:
precision 0.48834443387250237
recall 0.4835138954309939
f1 0.4859171597633136
19.260887 Seconds taken
~  Token is so important to keep around, my gosh

-UPPPER
SCORES:
precision 0.8827220077220077
recall 0.8615167216203485
f1 0.8719904648390941
20.665010000000002 Seconds taken
~ Slight improvement

-Title
SCORES:
precision 0.8831858407079646
recall 0.8226566179934055
f1 0.8518473356907693
20.977396 Seconds taken
~ Finally, one that might be worth it.

-InitTitle
SCORES:
precision 0.8644705882352941
recall 0.8652849740932642
f1 0.8648775894538605
20.566845 Seconds taken

-Punc
SCORES:
precision 0.8710056390977443
recall 0.8730569948186528
f1 0.8720301105622205
18.490571 Seconds taken
~ IMPROVEMENT

-Digit
SCORES:
precision 0.8849704142011834
recall 0.8805934997644842
f1 0.8827765316963758
21.679573 Seconds taken
~ IMPROVEMENT

It looks like tokens is FAR and away the best feature. It's the only one that really hurts the model on its own being removed. SO I went ahead and did one where ONLY token was on.

SCORES:
precision 0.8412263210368893
recall 0.7948657560056523
f1 0.8173891983531122
15.137909 Seconds taken

That's better than the all-but-token model. But still clearly not as good as adding more. This one probably misses some title-case hints, so let's turn that on.

SCORES:
precision 0.8666182522391672
recall 0.8431464908148846
f1 0.8547212605944848
15.680457 Seconds taken

Again, noticeably better (less than the 0.2 cutoff, but most of my results are)

I'm going with:

TokenFeature (duh)
TitlecaseFeature
InitialTitlecaseFeature (since it seems an obvious pair with previous)
WordShapeFeature (subsumes the role of DigitFeature at least)

SCORES:
precision 0.8863955119214586
recall 0.8930758360810175
f1 0.8897231346785548
40.609203 Seconds taken

The other ones should FEEL BAD








    """


def performance_analysis_comments2() -> None:
    """
    TODO: Describe your findings here
    On to the word vectors!

1.0
SCORES:
precision 0.9099648300117233
recall 0.9140367404616109
f1 0.9119962401597931
312.24639299999996 Seconds taken

Singing Seconds Batman, that's a lot of time! And only a minute of that is because of my weirdo bug. But we still manage to finally crack the gorgeous 0.9 divider. Why did I start with 1.0 instead of 0.5? Because I forgot to specify and it's the default lol.

0.5
SCORES:
precision 0.9091122592766557
recall 0.9116815826660386
f1 0.9103951081843837
296.135722 Seconds taken

There seems to be some time tradeoff with these. Can't wait for 4.0. The uptrend from 0.5 to 1.0 is small.

2.0
SCORES:
precision 0.9155492825217596
recall 0.9166274140367404
f1 0.9160880310697894
310.111641 Seconds taken

Not much more time here. There's actually even more improvement from 1 to 2 than from 0.5 to 1. But that could easily just be noise. Still our highest water.

4.0
SCORES:
precision 0.9104161768163649
recall 0.9119170984455959
f1 0.9111660195317096
297.166258 Seconds taken

Yeah, we have hit the point of diminished returns. I'm going to go for 2.0. After that it starts to overshadow other features? Not sure.

FULL PATH
SCORES:
precision 0.9049188044245705
recall 0.9055581723975507
f1 0.9052383755150087
38.730106 Seconds taken

Much faster, also gets us above 0.9. Good sign.

ALL PREFIXES:
SCORES:
precision 0.9046270066100094
recall 0.9024964672633067
f1 0.9035604810186277
50.007552 Seconds taken

That didn't show noticeable improvement. Lots of bunk/extraneous features.

[8,12,16,20]
SCORES:
precision 0.9057628719886632
recall 0.9032030146019784
f1 0.9044811320754718
41.79597 Seconds taken

[2,4,8,16]
SCORES:
precision 0.9038642789820923
recall 0.9034385303815355
f1 0.9036513545347468
42.716829 Seconds taken

[4,6,10,20]
SCORES:
precision 0.9067076051015588
recall 0.9041450777202072
f1 0.9054245283018868
39.199172000000004 Seconds taken

Okay, the best one is the one with the weirdest spacing, [4,6,10,20] but not by much. And it was shortest to run, which counts for something.


AND THE GRAND FINALE
SCORES:
precision 0.9173728813559322
recall 0.9178049929345267
f1 0.9175888862726631
320.75494 Seconds taken

Heyyyy that's a better result than we've seen so far! What a good sign.

f1 for my best without them is only 0.8897231346785548


    """



class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        # extractor = WindowedTokenFeatureExtractor(
        #     [
        #         BiasFeature(),
        #         TokenFeature(),
        #         UppercaseFeature(),
        #         TitlecaseFeature(),
        #         InitialTitlecaseFeature(),
        #         PunctuationFeature(),
        #         DigitFeature(),
        #         WordShapeFeature(),
        #     ],
        #     1,
        # )
        self.window = window_size
        self.feature_extractors = feature_extractors

    # Pump out a list of dictionaries
    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        to_return = []
        for current_idx in range(len(tokens)):
            to_append = {}
            for extractor in self.feature_extractors:
                for relative_idx in range(- self.window, self.window + 1):
                    if len(tokens) > current_idx + relative_idx > -1:
                        extractor.extract(tokens[current_idx + relative_idx], current_idx, relative_idx, tokens, to_append)
            to_return.append(to_append)
        return to_return










# HW2 stuff below

def load_conll2003(path: str, nlp: Language) -> List[Doc]:
    with open(path, encoding="utf8") as corpus:
        return list(CoNLLIngester().extract_docs(corpus, nlp))

def spacy_doc_from_sentences(
    sentences: List[List[str]], labels: List[str], nlp: Language
) -> Doc:
    # Create initial doc
    all_tokens = list(chain.from_iterable(sentences))
    # Mark that every token is followed by space
    spaces = [True] * len(all_tokens)
    doc = Doc(nlp.vocab, words=all_tokens, spaces=spaces)

    # Set sentence boundaries
    tok_idx = 0
    for sentence in sentences:
        for sentence_idx in range(len(sentence)):
            # First token should have start to True, all others False
            doc[tok_idx].is_sent_start = sentence_idx == 0
            tok_idx += 1

    if labels:
        if len(labels) != len(all_tokens):
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match number of tokens ({len(all_tokens)})"
            )

        # Create entities after converting IOB (actually BIO) to BILUO
        doc.ents = spans_from_biluo_tags(doc, iob_to_biluo(labels))

    return doc


UPPERCASE_RE = regex.compile(r"[\p{Lu}\p{Lt}]")
LOWERCASE_RE = regex.compile(r"\p{Ll}")
DIGIT_RE = re.compile(r"\d")

PUNC_REPEAT_RE = regex.compile(r"\p{P}+")


# class FeatureExtractor(ABC):
#     @abstractmethod
#     def extract(
#         self,
#         token: str,
#         current_idx: int,
#         relative_idx: int,
#         tokens: Sequence[str],
#         features: Dict[str, float],
#     ) -> None:
#         raise NotImplementedError


class EntityEncoder(ABC):
    @abstractmethod
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        raise NotImplementedError

class CoNLLIngester:
    def __init__(self) -> None:
        self.sentences: List[List[str]] = []
        self.tokens: List[str] = []
        self.labels: List[str] = []

    def reset(self) -> None:
        self.sentences = []
        self.tokens = []
        self.labels = []

    def _create_document(self, nlp: Language) -> Doc:
        assert self.sentences, "No sentences to create document with"
        doc = spacy_doc_from_sentences(self.sentences, self.labels, nlp)
        self.reset()
        return doc

    def extract_docs(self, corpus: TextIO, nlp: Language) -> Generator[Doc, None, None]:
        for line in corpus:
            if line.startswith("-DOCSTART-"):
                # Add the last sentence if needed
                if self.tokens:
                    self.sentences.append(self.tokens)

                # Create a document if there are sentences, otherwise we are at the first
                # docstart in the corpus and there's nothing to create
                if self.sentences:
                    yield self._create_document(nlp)
            elif line.strip():
                # Sample line:
                # German JJ B-NP B-MISC
                fields = line.split()
                self.tokens.append(fields[0])
                self.labels.append(fields[-1])
            else:
                # End of sentence if there are tokens, otherwise this is a blank leading
                # line before the first sentence and there's nothing to do.
                if self.tokens:
                    self.sentences.append(self.tokens)
                    self.tokens = []

        # Finish off document
        if self.tokens:
            self.sentences.append(self.tokens)
        yield self._create_document(nlp)


class PRF1(NamedTuple):
    precision: float
    recall: float
    f1: float


def span_prf1(
        reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> Dict[str, PRF1]:
    decoded_gold1 = [doc.ents for doc in reference_docs]
    decoded_gold = []
    for ents in decoded_gold1:
        decoded_gold.extend(ents)
    decoded_prediction1 = [doc.ents for doc in test_docs]
    decoded_prediction = []
    for ents in decoded_prediction1:
        decoded_prediction.extend(ents)
    cur_pred = 0
    cur_gold = 0

    true_pos = []
    false_pos = []
    false_neg = []

    for _ in range(len(decoded_gold) + len(decoded_prediction)): # this is the upper bound for iterating
        # Do something akin to merge sort. Iterate through each list, 'popping' off the one earlier in docs
        # Could probably refactor to actually use popping and holding current values being considered
        # Wouldn't save a lot of lines though

        if cur_pred >= len(decoded_prediction):
            # put the rest into the false negative bin and stop iterating
            false_neg.extend(decoded_gold[cur_gold:])
            break

        elif cur_gold >= len(decoded_gold):
            # put the rest in the false positive bin
            false_pos.extend(decoded_prediction[cur_pred:])
            break

        elif decoded_prediction[cur_pred].start == decoded_gold[cur_pred].start and \
                decoded_prediction[cur_pred].end == decoded_gold[cur_pred].end:
            if typed and decoded_prediction[cur_pred].label_ != decoded_gold[cur_pred].label_:
                # False pos and false neg
                false_pos.append(decoded_prediction[cur_pred])
                false_neg.append(decoded_gold[cur_gold])
            else:
                # True positive, only time it'll actually be appended to!
                true_pos.append(decoded_prediction[cur_pred])
            # Increment both regardless.
            cur_pred += 1
            cur_gold += 1

        else:
            # Not a match on length.
            if decoded_prediction[cur_pred].start < decoded_gold[cur_gold].start:
                # false positive
                false_pos.append(decoded_prediction[cur_pred])
                cur_pred += 1
            elif decoded_prediction[cur_pred].start > decoded_gold[cur_gold].start:
                # false negative
                false_neg.append(decoded_gold[cur_gold])
                cur_gold += 1
            else:
                #Obvs neither matches
                false_pos.append(decoded_prediction[cur_pred])
                false_neg.append(decoded_gold[cur_gold])

                cur_gold += 1
                cur_pred += 1

    # NOW go through each of the three lists and count it all up.
    to_return = {}
    types = set()
    types.add("")  # Look at this lifehack

    if typed:
        for tag in decoded_gold:
            types.add(tag.label_)
        for tag in decoded_prediction:
            types.add(tag.label_)
        #print(types)
    for label in types:
        # It's still O(N)
        tp = len([x for x in true_pos if label in x.label_]) # "" is in everything
        fp = len([x for x in false_pos if label in x.label_])
        fn = len([x for x in false_neg if label in x.label_])
        if (tp == 0 and fp == 0):
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp == 0 and fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 / (1.0 / precision + 1.0 / recall)  # Harmonic mean woo
        to_return[label] = PRF1(precision, recall, f1)

    return to_return


class BiasFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if relative_idx == 0:
            features["bias"] = 1.0


class TokenFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        features["tok[" + str(relative_idx) + "]=" + token] = 1.0


class UppercaseFeature(FeatureExtractor):
    # TODO: Implement
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if token.isupper():
            features["uppercase[" + str(relative_idx) + "]"] = 1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if token.istitle():  # ??? Not sure.
            features["initialtitlecase[" + str(relative_idx) + "]"] = 1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if token.istitle() and (current_idx + relative_idx == 0
                                or tokens[current_idx + relative_idx - 1] in ".!?"):  # Preceding token ends sentence
            features["titlecase[" + str(relative_idx) + "]"] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if PUNC_REPEAT_RE.match(token):
            features["punc[" + str(relative_idx) + "]"] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        if DIGIT_RE.match(token):
            features["digit[" + str(relative_idx) + "]"] = 1.0


class WordShapeFeature(FeatureExtractor):
    def extract(
            self,
            token: str,
            current_idx: int,
            relative_idx: int,
            tokens: Sequence[str],
            features: Dict[str, float], ):
        shape = regex.sub(UPPERCASE_RE, 'X',
                          regex.sub(LOWERCASE_RE, 'x',
                                    re.sub(DIGIT_RE, '0', token)))
        features["shape[" + str(relative_idx) + "]=" + shape] = 1.0

#
# class WindowedTokenFeatureExtractor:
#     def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
#         # extractor = WindowedTokenFeatureExtractor(
#         #     [
#         #         BiasFeature(),
#         #         TokenFeature(),
#         #         UppercaseFeature(),
#         #         TitlecaseFeature(),
#         #         InitialTitlecaseFeature(),
#         #         PunctuationFeature(),
#         #         DigitFeature(),
#         #         WordShapeFeature(),
#         #     ],
#         #     1,
#         # )
#         self.window = window_size
#         self.feature_extractors = feature_extractors
#
#     # Pump out a list of dictionaries
#     def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
#         to_return = []
#         for current_idx in range(len(tokens)):
#             to_append = {}
#             for extractor in self.feature_extractors:
#                 for relative_idx in range(- self.window, self.window + 1):
#                     if len(tokens) > current_idx + relative_idx > -1:
#                         extractor.extract(tokens[current_idx + relative_idx], current_idx, relative_idx, tokens, to_append)
#             to_return.append(to_append)
#         return to_return


class CRFsuiteEntityRecognizer:
    def __init__(
            self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        # TODO: Implement
        self._encoder = encoder
        self.fe = feature_extractor #Iron? Close enough
        self.trainer = None
        self.tagger = None
        pass

    @property
    def encoder(self) -> EntityEncoder:
        return self._encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        # TODO: Implement
        self.trainer = pycrfsuite.Trainer()
        self.trainer.set_params(params)
        self.trainer.select(algorithm)
        for doc in docs:
            for sent in doc.sents:
                # Eh, I nest a loop through three times. Sue me.
                toks = []
                for tok in list(sent):  # next(doc.sents)
                    toks.append(str(tok))
                feats = self.fe.extract(toks)
                bilou_ents = self.encoder.encode(sent) # Hey, this is where to use the encoder lol
                self.trainer.append(feats, bilou_ents)
        self.trainer.train(path)
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        # TODO: Implement
        if self.trainer == None:
            # self.trainer not yet initialized by train()
            raise ValueError
        entities = []
        for sent in doc.sents:
            toks = []
            for tok in list(sent):  # next(doc.sents)
                toks.append(str(tok))
            tags = self.predict_labels(toks)
            to_append = decode_bilou(tags, sent, doc)
            entities.extend(to_append)
        doc.ents = entities
        #print(doc,"\n", "BANANA", "\n", entities)
        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        feats = self.fe.extract(tokens)
        tags = self.tagger.tag(feats)

        return tags


class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        # [George, Washington, was, president, of, the, United, States, of, America, .]
        #return ['B-PER', 'L-PER', 'O', 'O', 'O', 'O', 'B-GPE', 'I-GPE', 'I-GPE', 'L-GPE', 'O']

        # noticing this was an attribute is what cost me the two days
        # print(tokens[0].ent_iob_)
        # If we're supposed to not use that... then I can't imagine what it was supposed to be.

        bio = [tok.ent_iob_ for tok in tokens]
        ent_types = [tok.ent_type_ for tok in tokens]
        bilou = bio_to_bilou(bio)

        conjoined = [bilou[x] + "-" + ent_types[x] if bilou[x] != "O" else bilou[x] for x in range(len(tokens))]
        return conjoined


class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        bio = [tok.ent_iob_ for tok in tokens]
        ent_types = [tok.ent_type_ for tok in tokens]
        conjoined = [bio[x] + "-" + ent_types[x] if bio[x] != "" else "O" for x in range(len(tokens))]
        return conjoined


class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        bio = [tok.ent_iob_ for tok in tokens]
        ent_types = [tok.ent_type_ for tok in tokens]
        conjoined = ["I-" + ent_types[x] if bio[x] != "" else "O" for x in range(len(tokens))]
        return conjoined

def bio_to_bilou(labels: Sequence[str]) -> List[str]:
    # TODO: Implement
    to_return = []
    # B, O or I
    last = "O"
    # There are only so many possibilities, best to just label them all IMO
    # Invalid ones get turned into more valid tags, not that it matters
    switcher = {
        'BBB': "U", 'BOB': "O", 'BIB': "L",
        'OBB': "U", 'OOB': "O", 'OIB': "U",
        'IBB': "U", 'IOB': "O", 'IIB': "L",

        'BBO': "U", 'BOO': "O", 'BIO': "L",
        'OBO': "U", 'OOO': "O", 'OIO': "U",
        'IBO': "U", 'IOO': "O", 'IIO': "L",

        'BBI': "B", 'BOI': "O", 'BII': "I",
        'OBI': "B", 'OOI': "O", 'OII': "B",
        'IBI': "B", 'IOI': "O", 'III': "I"
    }
    for x in range(len(labels)):
        if len(labels[x]) == 0:
            labels[x] = "O" # Handling empty cases.
    for x in range(len(labels)):
        bio = labels[x][0]
        if x == 0:
            last = 'O'
        else:
            last = labels[x - 1][0]
        if x == len(labels) - 1:
            next = 'O'
        else:
            next = labels[x + 1][0]
        to_return.append(switcher[last + bio + next] + labels[x][1:])
    return(to_return)

def add_ent(labels: Sequence[str], tokens: Sequence[Token], doc: Doc, cur_ent: List[int], spanlist):
    # Make an entity. This is because an O, B, or U has occurred.
    # Put in some checks for span validity and such. This is mostly just a helper function
    if cur_ent[0] >= cur_ent[1]:
        return
    # Get position in document from position in sentence
    first = tokens[cur_ent[0]].i
    second = first + (cur_ent[1] - cur_ent[0])
    spanlist.append(Span(doc, first, second, labels[cur_ent[0]][2:]))

def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    to_return = []  # List of spaCy tokens
    cur_ent = [len(labels), 0]
    last_label = ""
    for lab in range(len(labels)):
        if labels[lab][0] in "BU":
            # Effectively the same thing, with how we handle improper tags
            add_ent(labels, tokens, doc, cur_ent, to_return)
            cur_ent[0], cur_ent[1] = lab, lab
            last_label = labels[lab][2:]
        elif labels[lab][0] in "IL":
            # Only care if it's SUPPOSED to be a B, or if the entity type has changed
            if cur_ent[0] == len(labels) or last_label != labels[lab][2:]:
                # replicate code for B/U
                add_ent(labels, tokens, doc, cur_ent, to_return)
                cur_ent[0], cur_ent[1] = lab, lab
            last_label = labels[lab][2:]
        elif labels[lab][0] == "O":
            # cur_start = len(labels) # Since not an entity, make it impossible to go in range
            last_label = ""
            add_ent(labels, tokens, doc, cur_ent, to_return)
            cur_ent[0], cur_ent[1] = len(labels), lab
        cur_ent[1] += 1
    add_ent(labels, tokens, doc, cur_ent, to_return)  # In case sentence ends on BIL
    return to_return




if __name__ == "__main__":
    docs = []
    docs2 = [] # Look, my thing edits documents in place, it's easier to just have two.
    with open("batch_9_hutchens.jsonl", "r") as to_read:
        #print(to_read.read()[:1000])
        d_list = [json.loads(line) for line in to_read.readlines()]
        NLP = spacy.load("en_core_web_sm", disable=["ner"])

        for d in d_list:
            try:
                docs.append(ingest_json_document(d, NLP))
                docs2.append(ingest_json_document(d, NLP))
            except ValueError:
                print("Problem in document (probably the French one): ", d)

    print(len(docs))
    train = docs[:199]
    test = docs[199:]
    test2 = docs2[199:]

    tally = 0
    for doc in docs:
        tally += len(doc.ents)
    print(tally)
    # split on stuff.

    #
    encoder = "bilou"
    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            [
                BrownClusterFeature("./data/restaurant_reviews_all_truecase_paths", use_prefixes=True, prefixes = [2,4,8,16]),
                WordVectorFeature(vectors_path="./data/restaurant_reviews_all_truecase.magnitude", scaling=1.0),
                # BiasFeature(),
                TokenFeature(),
                #UppercaseFeature(),
                TitlecaseFeature(),
                InitialTitlecaseFeature(),
                PunctuationFeature(),
                DigitFeature(),
                WordShapeFeature(),
            ],
            1,  # Window size
        ),
        {"bilou": BILOUEncoder(),
         "bio": BIOEncoder(),
         "io": IOEncoder()}[encoder],
    )

    crf.train(train, "ap", {"max_iterations": 40}, "tmp.model")
    test_predictions = [crf(doc) for doc in test]

    labels = ["", "BIZ", "QUAL", "DISH-INGRED", "SERV"]
    print("RESULTS???? : \n")
    result = span_prf1_type_map(test2, test_predictions, type_map={"DISH":"DISH-INGRED", "INGRED":"DISH-INGRED"})
    for label in labels:
        print(label + ":" , str(result[label]))

"""
API at https://radimrehurek.com/gensim/models/word2vec.html
tutorial at https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296


"""

import json
from gensim.models import Word2Vec
import re

train_set = {}

# get only train set (first 565 documents)
with open("annotated_files/annotated_data.jsonl", encoding="utf-8") as f:
    decoder = json.JSONDecoder()
    for i,document in enumerate(f):
        if i > 565:
            break
        document = decoder.decode(document)
        train_set[i] = document

train_set_split = []

for doc in train_set:
    sentenced = re.split(r'[.?!]', train_set[doc]["text"])
    for sentence in sentenced:
        sentence = re.findall(r"[\w']+|[\[\]\(\):,;]", sentence)
        sentence.extend(".")
        train_set_split.append(sentence)

EMB_DIM = 300

# w2v = Word2Vec(train_set_split, size=EMB_DIM, window=4, min_count=3, negative=15, iter=20)

# word_vectors = w2v.wv
# result = word_vectors.similar_by_word("Picard")
# print(result)

# w2v.save('trek_w2v.model')

new_model = Word2Vec.load('trek_w2v.model')
new_word_vectors = new_model.wv
result = new_word_vectors.similar_by_word("Picard")
print(result)

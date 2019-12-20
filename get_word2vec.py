"""
API at https://radimrehurek.com/gensim/models/word2vec.html
tutorial at https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296


"""
from gensim.models import Word2Vec
import spacy
import os

train_set = []

# get entire corpus
rootdir = './Datasets'
corpus = []
nlp = spacy.load("en_core_web_sm", disable=["ner"])
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            with open(os.path.join(subdir, file), 'r') as f:
                lines = f.readlines()
                text = "".join(lines[5:])
                text = text.lower()
                text = text.replace(")[", ") ")
                text = text.replace("\n", " ")
                text = text.replace(".)", ". ")
                doc = nlp(text)
                sentences = doc.sents
                for sentence in sentences:
                    corpus.append([str(token) for token in list(sentence.__iter__())])

w2v = Word2Vec(corpus, size=300, window=4, min_count=3, negative=15, iter=20)

word_vectors = w2v.wv
result = word_vectors.similar_by_word("picard")
print(result)
w2v.save('trek_w2v.model')

#odel = Word2Vec.load('trek_w2v_fics.model')
#vectors = model.wv
#print(vectors.similar_by_word("vulcan"))



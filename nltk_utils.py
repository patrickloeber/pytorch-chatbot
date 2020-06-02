import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# stemming examples
# print(stemmer.stem("maximum"), stemmer.stem("saying"), stemmer.stem("crying"))
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """return bag of words array: 
    1 for each known word that exists in the sentence, 0 otherwise
    """
    # stem each word
    sentence_words = [self.stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
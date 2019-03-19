import random
from abc import ABCMeta, abstractmethod

from nltk.corpus import brown


def sentence_ngram_iterator(sentence, order):
    for idx in range(len(sentence) - (order - 1)):
        yield tuple(sentence[idx:idx + order])


def sentence_word_iterator(sentence):
    for word in sentence:
        yield word


def expand_tokenized_sentence(sentence, start_token=None, end_token=None):
    if start_token:
        sentence = [start_token] + sentence
    if end_token:
        sentence.append(end_token)
    return sentence


class CorpusReaderInterface(object, metaclass=ABCMeta):

    def __init__(self, dictionary=None, start_token=None, end_token=None):
        self.dictionary = dictionary
        self._start_token = start_token
        self._end_token = end_token

    @property
    def start_token(self):
        return self._start_token

    @property
    def end_token(self):
        return self._end_token

    @abstractmethod
    def sentence_iterator(self):
        raise NotImplementedError("subclasses must override this method")

    def ngram_iterator(self, order):
        assert order > 0

        if order > 1:
            def ngram_iterator(sent):
                return sentence_ngram_iterator(sent, order)
        else:
            ngram_iterator = sentence_word_iterator

        for sentence in self.sentence_iterator():
            for ngram in ngram_iterator(sentence):
                yield ngram

    @staticmethod
    @abstractmethod
    def get_random_sentence():
        raise NotImplementedError("subclasses must override this method")


class BrownCorpusReader(CorpusReaderInterface):

    def sentence_iterator(self):
        for doc_id in brown.fileids():
            sentences = brown.sents(doc_id)
            for sent in sentences:
                sent = [word.lower() for word in sent]
                sent = expand_tokenized_sentence(sent, self._start_token, self._end_token)
                if self.dictionary is not None:
                    sent = [self.dictionary[word] for word in sent]
                yield sent

    @staticmethod
    def get_random_sentence():
        fid = random.sample(brown.fileids(), 1)[0]
        sents = brown.sents(fid)
        sid = random.sample(range(len(sents)), 1)[0]
        return sents[sid]

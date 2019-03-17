import random
from abc import ABCMeta, abstractmethod

from nltk.corpus import brown


class CorpusReaderInterface(object, metaclass=ABCMeta):

    def __init__(self, dictionary=None, start_token=None, end_token=None):
        self._dictionary = dictionary
        self._start_token = start_token
        self._end_token = end_token

    @property
    def dictionary(self):
        return self._dictionary

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
        sentence_iterator = self._make_sentence_iterator(order)
        for sent in self.sentence_iterator():
            for ngram in sentence_iterator(sent):
                yield ngram

    def _make_sentence_iterator(self, ngram_order):
        if ngram_order == 1:
            return self._sentence_word_iterator

        def sent_it(sentence):
            return self._sentence_ngram_iterator(sentence, ngram_order)
        return sent_it

    @staticmethod
    def _sentence_word_iterator(sentence):
        for word in sentence:
            yield word

    @staticmethod
    def _sentence_ngram_iterator(sentence, order):
        for idx in range(len(sentence) - (order - 1)):
            yield tuple(sentence[idx:idx + order])

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
                if self._start_token:
                    sent = [self._start_token] + sent
                if self._end_token:
                    sent.append(self._end_token)
                if self._dictionary is not None:
                    sent = [self._dictionary[word] for word in sent]
                yield sent

    @staticmethod
    def get_random_sentence():
        fid = random.sample(brown.fileids(), 1)[0]
        sents = brown.sents(fid)
        sid = random.sample(range(len(sents)), 1)[0]
        return sents[sid]

import pickle
import random
from collections.__init__ import Counter
from math import log

from tools.corpus_readers import expand_tokenized_sentence, sentence_ngram_iterator


class NeuralNgramModel(object):
    def __init__(self, corpus_reader, order=2):
        self._order = order
        self._corpus_reader = corpus_reader

        self._vocab = corpus_reader.dictionary
        self._start_token = corpus_reader.start_token
        self._end_token = corpus_reader.end_token

    def train(self, learning_rate=0.01):
        ngram_ids = [ids for ids in self._corpus_reader.ngram_iterator(2)]
        X_ids, Y_ids = zip(*ngram_ids)

        print("done")


class MarkovNgramModel(object):

    def __init__(self, corpus_reader, order=2, smoothing=0.01):
        self._order = order
        self._corpus_reader = corpus_reader

        self._vocab = corpus_reader.dictionary
        self._start_token = corpus_reader.start_token
        self._end_token = corpus_reader.end_token

        self._corpus_size = None
        self._ngram_counters = None
        self.smoothing = smoothing
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters = dict()
        for n in range(self._order):
            self._ngram_counters[n + 1] = Counter(self._corpus_reader.ngram_iterator(n + 1))

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise RuntimeError("{} is not a {} instance".format(filepath, cls.__name__))
        return model

    @property
    def order(self):
        return self._order

    @property
    def unigram_counts(self):
        return self.get_ngram_counts(1)

    def get_ngram_counts(self, order):
        return self._ngram_counters[order]

    @property
    def start_token(self):
        return self._start_token

    @property
    def end_token(self):
        return self._end_token

    @property
    def vocab(self):
        return self._vocab

    @property
    def corpus_size(self):
        if self._corpus_size is None:
            total_count = sum(self.unigram_counts.values())
            total_count -= self.unigram_counts.get(self._start_token, 0) + self.unigram_counts.get(self._end_token, 0)
            self._corpus_size = total_count
        return self._corpus_size

    def word_prob(self, word):
        assert isinstance(word, str)
        # word = word.lower()
        if word not in self.vocab:
            raise ValueError(word, ' not in vocab')

        prob = self.unigram_counts[self._vocab[word]] / self.corpus_size
        return prob

    def bigram_prob(self, bigram):
        assert len(bigram) == 2
        # bigram = tuple([word.lower() for word in bigram])  # lowercase bigram

        # translate bigram into token_ids
        bigram = tuple(self.vocab[word] for word in bigram)

        # use smoothing
        prob = (self.get_ngram_counts(2)[bigram] + self.smoothing) / \
               (self.unigram_counts[bigram[0]] + self.smoothing * len(self.vocab))
        return prob

    def sentence_log_prob(self, sentence):
        length = len(sentence)
        if length < 2:
            raise ValueError("sentence must be at least of length 2")

        sentence = [word.lower() for word in sentence]
        sentence = expand_tokenized_sentence(sentence, self._start_token, self._end_token)

        # we don't care about the overall probability for a sentence to start with a particular word,
        # especially if we are using a start token (that takes care of start probability via bigram prob of
        # [start, 1st word])
        logprob = 0
        ct = 0
        for bigram in sentence_ngram_iterator(sentence, 2):
            ct += 1
            # print(ct, bigram)
            prob = self.bigram_prob(bigram)
            # if smoothing = 0, some ngrams can have 0 probability if they never appeared in the corpus. In that case
            # the sentence has 0 probability -> return logprob = -inf
            if prob == 0.0:
                return -float('inf')
            logprob += log(prob)

        logprob /= ct
        return logprob


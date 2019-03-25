import pickle

import numpy

from abc import ABCMeta, abstractmethod
from collections.__init__ import Counter, defaultdict
from math import log

from numpy import mean

from tools.corpus_readers import expand_tokenized_sentence, sentence_ngram_iterator
from tools.math_utilities import softmax


class NgramModelInterface(object, metaclass=ABCMeta):
    def __init__(self, corpus_reader, order=2):
        self._order = order
        self._corpus_reader = corpus_reader
        self._corpus_size = None

        self._vocab = corpus_reader.dictionary
        self._start_token = corpus_reader.start_token
        self._end_token = corpus_reader.end_token

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
    def start_token(self):
        return self._start_token

    @property
    def end_token(self):
        return self._end_token

    @property
    def vocab(self):
        return self._vocab

    @abstractmethod
    def sentence_log_prob(self, sentence):
        raise NotImplementedError("subclasses must override this method")


class NeuralNgramModel(NgramModelInterface):

    def __init__(self, corpus_reader, order=2):
        super().__init__(corpus_reader, order)
        self._weights = None

    def train(self, learning_rate, batch_size, epochs):

        X_ids, Y_ids = zip(*self._corpus_reader.ngram_iterator(2))
        X_ids = numpy.array(X_ids)
        Y_ids = numpy.array(Y_ids)

        X_ids_T = defaultdict(list)
        for i, x in enumerate(X_ids):
            X_ids_T[x].append(i)

        V = len(self._vocab)
        N = len(X_ids)

        print(N)
        print(V)

        self._weights = numpy.random.randn(V, V) / V**2

        # initial cost function evaluation
        # Y_pred = softmax(self._weights[X_ids])
        # log_probs = - numpy.log(numpy.array([Y_pred[i, Y_ids[i]] for i in range(N)]))
        # J = log_probs.sum() / N
        # print(J)

        for epoch in range(epochs):
            print(80 * "=")
            print("epoch {}/{}".format(epoch, epochs))
            print()

            sample_inds = numpy.arange(0, N)
            numpy.random.shuffle(sample_inds)

            batches = [sample_inds[ind:ind + batch_size] for ind in range(0, N, batch_size)]
            num_batches = len(batches)
            for n_b, batch in enumerate(batches):

                curr_batch_size = len(batch)

                X = numpy.array(X_ids[batch])
                Y = numpy.array(Y_ids[batch])

                # forward pass
                Z = self._weights[X]
                Y_pred = softmax(Z)

                log_probs = - numpy.log(numpy.array([Y_pred[i, j_i] for i, j_i in enumerate(Y)]))
                J = mean(log_probs)
                print("batch {}/{}, J={}".format(n_b, num_batches, J))

                for i, j_i in enumerate(Y):
                    # Y_pred <- Y_pred - Y
                    Y_pred[i, j_i] -= 1


            # print(len(batches))
            # print([len(batch) for batch in batches])

    def _forward_pass(self, X_ids):
        Z = self._weights[X_ids]

    def sentence_log_prob(self, sentence):
        return 1


class MarkovNgramModel(NgramModelInterface):

    def __init__(self, corpus_reader, order=2, smoothing=0.01):
        super().__init__(corpus_reader, order)

        self._ngram_counters = None
        self.smoothing = smoothing
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters = dict()
        for n in range(self._order):
            self._ngram_counters[n + 1] = Counter(self._corpus_reader.ngram_iterator(n + 1))

    @property
    def corpus_size(self):
        if self._corpus_size is None:
            total_count = sum(self.unigram_counts.values())
            total_count -= self.unigram_counts.get(self._start_token, 0) + self.unigram_counts.get(self._end_token, 0)
            self._corpus_size = total_count
        return self._corpus_size

    @property
    def unigram_counts(self):
        return self.get_ngram_counts(1)

    def get_ngram_counts(self, order):
        return self._ngram_counters[order]

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


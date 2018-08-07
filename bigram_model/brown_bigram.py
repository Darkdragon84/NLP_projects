import random
from math import log
from nltk.corpus import brown
from collections import Counter


def brown_sentence_iterator():
    for doc_id in brown.fileids():
        sentences = brown.sents(doc_id)
        for sent in sentences:
            yield sent


def brown_word_iterator():
    sit = brown_sentence_iterator()
    for sent in sit:
        for word in sent:
            yield word.lower()


def brown_ngram_iterator(order):
    if order < 2:
        raise ValueError("order must be at least 2")

    for sent in brown_sentence_iterator():
        sent = [word.lower() for word in sent]
        for ngram in sentence_ngram_iterator(sent, order):
            yield ngram


def sentence_ngram_iterator(sentence, order):
    for idx in range(len(sentence) - (order - 1)):
        yield tuple(sentence[idx:idx + order])


def get_random_brown_sentence():
    fid = random.sample(brown.fileids(), 1)[0]
    sents = brown.sents(fid)
    sid = random.sample(range(len(sents)), 1)[0]
    return sents[sid]


class BrownNgramModel(object):

    def __init__(self, order=2):
        self._order = order
        self._vocab = None
        self._vocab_size = None
        self._corpus_size = None
        self._ngram_counters = None
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters = dict()
        self._ngram_counters[1] = Counter(brown_word_iterator())
        for n in range(1, self._order):
            self._ngram_counters[n+1] = Counter(brown_ngram_iterator(n+1))

    @property
    def order(self):
        return self._order

    @property
    def unigram_counts(self):
        return self.get_ngram_counts(1)

    def get_ngram_counts(self, order):
        return self._ngram_counters[order]

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = set(self.unigram_counts.keys())
        return self._vocab

    @property
    def vocab_size(self):
        if self._vocab_size is None:
            self._vocab_size = len(self.vocab)
        return self._vocab_size

    @property
    def corpus_size(self):
        if self._corpus_size is None:
            ct = 0
            for n in self.unigram_counts.values():
                ct += n
            self._corpus_size = ct
        return self._corpus_size

    def word_log_prob(self, word):
        assert isinstance(word, str)
        word = word.lower()
        if word not in self.vocab:
            raise ValueError(word, ' not in vocab')

        logprob = log(self.unigram_counts[word]) - log(self.corpus_size)
        return logprob

    def bigram_log_prob(self, bigram):
        assert len(bigram) == 2
        bigram = tuple([word.lower() for word in bigram])  # lowercase bigram

        if not set(bigram).issubset(self.vocab):
            raise ValueError(bigram, ' not in vocab')

        # use one smoothing
        logprob = log(self.get_ngram_counts(2)[bigram] + 1) - log(self.unigram_counts[bigram[0]] + self.vocab_size)
        return logprob

    def sentence_log_prob(self, sentence):
        length = len(sentence)
        assert length > 2

        sentence = [word.lower() for word in sentence]
        if not set(sentence).issubset(self.vocab):
            raise ValueError(sentence, ' not in vocab')

        logprob = self.word_log_prob(sentence[0])
        # for idx in range(length - (self._order - 1)):
        #     bigram = tuple(sentence[idx:idx + self._order])
        for bigram in sentence_ngram_iterator(sentence, 2):
            logprob += self.bigram_log_prob(bigram)

        logprob /= length
        return logprob


def main():
    model = BrownNgramModel(2)

    sent = get_random_brown_sentence()
    print(sent)
    print(model.sentence_log_prob(sent))


if __name__ == '__main__':
    main()

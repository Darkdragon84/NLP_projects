import random
from math import log
from nltk.corpus import brown
from collections import Counter

from tools.dictionary import Dictionary


def brown_sentence_iterator(dictionary=None):
    for doc_id in brown.fileids():
        sentences = brown.sents(doc_id)
        for sent in sentences:
            sent = [word.lower() for word in sent]
            if dictionary is not None:
                dictionary.add_tokens(sent)
                sent = [dictionary[word] for word in sent]
            yield sent


def brown_word_iterator(dictionary=None):
    sit = brown_sentence_iterator(dictionary)
    for sent in sit:
        for word in sent:
            yield word


def brown_ngram_iterator(order, dictionary=None):
    if order < 2:
        raise ValueError("order must be at least 2")

    for sent in brown_sentence_iterator(dictionary):
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

        self._vocab = Dictionary()
        self._vocab_size = None
        self._vocab_as_list = None

        self._corpus_size = None
        self._ngram_counters = None
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters = dict()
        self._ngram_counters[1] = Counter(brown_word_iterator(self._vocab))
        for n in range(2, self._order + 1):
            self._ngram_counters[n] = Counter(brown_ngram_iterator(n, self._vocab))

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
        return self._vocab

    @property
    def corpus_size(self):
        if self._corpus_size is None:
            ct = 0
            for n in self.unigram_counts.values():
                ct += n
            self._corpus_size = ct
        return self._corpus_size

    def word_prob(self, word):
        assert isinstance(word, str)
        word = word.lower()
        if word not in self.vocab:
            raise ValueError(word, ' not in vocab')

        rob = self.unigram_counts[self.vocab[word]] / self.corpus_size
        return rob

    def bigram_prob(self, bigram, smoothing=1.):
        assert len(bigram) == 2
        bigram = tuple([word.lower() for word in bigram])  # lowercase bigram

        if bigram not in self.vocab:
            raise ValueError(bigram, ' not in vocab')

        # translate bigram into token_ids
        bigram = tuple(self.vocab[word] for word in bigram)

        # use smoothing
        prob = (self.get_ngram_counts(2)[bigram] + smoothing) / \
               (self.unigram_counts[bigram[0]] + smoothing * len(self.vocab))
        return prob

    def sentence_log_prob(self, sentence, smoothing=1.):
        length = len(sentence)
        assert length > 2

        sentence = [word.lower() for word in sentence]
        if sentence not in self.vocab:
            raise ValueError(sentence, ' not in vocab')


        logprob = log(self.word_prob(sentence[0]))
        # for idx in range(length - (self._order - 1)):
        #     bigram = tuple(sentence[idx:idx + self._order])
        for bigram in sentence_ngram_iterator(sentence, 2):
            prob = self.bigram_prob(bigram, smoothing)
            # if smoothing = 0, some ngrams can have 0 probability if they never appeared in the corpus. In that case
            # the sentence has 0 probability -> return logprob = -inf
            if prob == 0.0:
                return -float('inf')
            logprob += log(prob)

        logprob /= length
        return logprob

    def get_random_sentence_from_vocab(self, n_words):
        sentence = [random.choice(self._vocab.tokens) for _ in range(n_words)] + ['.']
        return sentence


def main():
    model = BrownNgramModel(2)
    smoothing = 0.5

    while True:
        sent1 = get_random_brown_sentence()
        sent2 = model.get_random_sentence_from_vocab(random.choice(range(5, 20)))

        logprob1 = model.sentence_log_prob(sent1, smoothing=smoothing)
        logprob2 = model.sentence_log_prob(sent2, smoothing=smoothing)

        print("{}: {}".format(logprob1, " ".join(sent1)))
        print("{}: {}".format(logprob2, " ".join(sent2)))

        choice = input("continue [Y/n]:")
        if choice == 'n':
            break


if __name__ == '__main__':
    main()

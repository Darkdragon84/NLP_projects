import random
from math import log
from nltk.corpus import brown
from collections import Counter

from tools.dictionary import Dictionary

START = 'START_TOKEN'
END = 'END_TOKEN'

TEST_SENTENCE = "They are trying to demonstrate some different ways of teaching and learning .".split()


def brown_sentence_iterator(dictionary=None, start_token=None, end_token=None):
    if dictionary is not None:
        delimiters = []
        if start_token:
            delimiters.append(start_token)
        if end_token:
            delimiters.append(end_token)
        dictionary.add_tokens(delimiters)

    for doc_id in brown.fileids():
        sentences = brown.sents(doc_id)
        for sent in sentences:
            sent = [word.lower() for word in sent]
            sent = expand_tokenized_sentence(sent, start_token, end_token)
            if dictionary is not None:
                dictionary.add_tokens(sent)
                sent = [dictionary[word] for word in sent]
            yield sent


def brown_ngram_iterator(order, dictionary=None, start_token=None, end_token=None):
    assert order > 0
    sentence_iterator = make_sentence_iterator(order)
    for sent in brown_sentence_iterator(dictionary, start_token, end_token):
        for ngram in sentence_iterator(sent):
            yield ngram


def make_sentence_iterator(ngram_order):
    if ngram_order == 1:
        return sentence_word_iterator

    def sit(sentence):
        return sentence_ngram_iterator(sentence, ngram_order)
    return sit


def sentence_word_iterator(sentence):
    for word in sentence:
        yield word


def sentence_ngram_iterator(sentence, order):
    for idx in range(len(sentence) - (order - 1)):
        yield tuple(sentence[idx:idx + order])


def get_random_brown_sentence():
    fid = random.sample(brown.fileids(), 1)[0]
    sents = brown.sents(fid)
    sid = random.sample(range(len(sents)), 1)[0]
    return sents[sid]


def expand_tokenized_sentence(tokens, start_token=None, end_token=None):
    if start_token:
        tokens = [start_token] + tokens
    if end_token:
        tokens = tokens + [end_token]
    return tokens


class BrownNgramModel(object):

    def __init__(self, order=2, start_token=None, end_token=None):
        self._order = order
        self._start_token = start_token
        self._end_token = end_token

        self._vocab = Dictionary()
        self._vocab_size = None
        self._vocab_as_list = None

        self._corpus_size = None
        self._ngram_counters = None
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters = dict()
        for n in range(self._order):
            self._ngram_counters[n + 1] = Counter(brown_ngram_iterator(n + 1, self._vocab,
                                                                       self._start_token, self._end_token))

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
            total_count = 0
            for token, count in self.unigram_counts.items():
                if token not in {self._start_token, self._end_token}:
                    total_count += count
            self._corpus_size = total_count
        return self._corpus_size

    def word_prob(self, word):
        assert isinstance(word, str)
        # word = word.lower()
        if word not in self.vocab:
            raise ValueError(word, ' not in vocab')

        rob = self.unigram_counts[self.vocab[word]] / self.corpus_size
        return rob

    def bigram_prob(self, bigram, smoothing=1.):
        assert len(bigram) == 2
        # bigram = tuple([word.lower() for word in bigram])  # lowercase bigram

        if bigram not in self.vocab:
            raise ValueError(bigram, ' not in vocab')

        # translate bigram into token_ids
        bigram = tuple(self.vocab[word] for word in bigram)

        # use smoothing
        prob = (self.get_ngram_counts(2)[bigram] + smoothing) / \
               (self.unigram_counts[bigram[0]] + smoothing * len(self.vocab))
        return prob

    def sentence_log_prob(self, sentence, smoothing=1.):

        # TODO how to properly measure the length in presence of start and end tokens?
        assert len(sentence) > 2

        sentence = [word.lower() for word in sentence]
        sentence = expand_tokenized_sentence(sentence, self._start_token, self._end_token)
        if sentence not in self.vocab:
            raise ValueError(sentence, ' not in vocab')

        # we don't care about the overall probability for a sentence to start with a particular word,
        # especially if we are using a start token
        logprob = 0
        ct = 0
        for bigram in sentence_ngram_iterator(sentence, 2):
            ct += 1
            print(ct, bigram)
            prob = self.bigram_prob(bigram, smoothing)
            # if smoothing = 0, some ngrams can have 0 probability if they never appeared in the corpus. In that case
            # the sentence has 0 probability -> return logprob = -inf
            if prob == 0.0:
                return -float('inf')
            logprob += log(prob)

        logprob /= ct
        return logprob

    def get_random_sentence_from_vocab(self, n_words):
        sentence = [random.choice(self._vocab.tokens) for _ in range(n_words)] + ['.']
        return sentence


def main():
    # model = BrownNgramModel(2)
    model = BrownNgramModel(2, START, END)
    smoothing = 0.1

    logprob = model.sentence_log_prob(TEST_SENTENCE, smoothing=smoothing)
    print("{}: {}".format(logprob, " ".join(TEST_SENTENCE)))

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

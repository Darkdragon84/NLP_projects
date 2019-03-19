import os
import pickle
import random
import warnings

import simplejson
from argparse import ArgumentParser
from math import log
from collections import Counter

from tools.corpus_readers import BrownCorpusReader, sentence_ngram_iterator, expand_tokenized_sentence
from tools.dictionary import Dictionary, make_and_save_dictionary

START = 'START_TOKEN'
END = 'END_TOKEN'

TEST_SENTENCE = "They are trying to demonstrate some different ways of teaching and learning .".split()


class BrownNgramModel(object):

    def __init__(self, corpus_reader, order=2):
        self._order = order
        self._corpus_reader = corpus_reader

        self._vocab = corpus_reader.dictionary
        self._start_token = corpus_reader.start_token
        self._end_token = corpus_reader.end_token

        self._corpus_size = None
        self._ngram_counters = None
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

    def bigram_prob(self, bigram, smoothing=1.):
        assert len(bigram) == 2
        # bigram = tuple([word.lower() for word in bigram])  # lowercase bigram

        # translate bigram into token_ids
        bigram = tuple(self.vocab[word] for word in bigram)

        # use smoothing
        prob = (self.get_ngram_counts(2)[bigram] + smoothing) / \
               (self.unigram_counts[bigram[0]] + smoothing * len(self.vocab))
        return prob

    def sentence_log_prob(self, sentence, smoothing=1.):
        length = len(sentence)
        if length < 2:
            raise ValueError("sentence must be at least of length 2")

        sentence = [word.lower() for word in sentence]
        sentence = expand_tokenized_sentence(sentence, self._start_token, self._end_token)

        # we don't care about the overall probability for a sentence to start with a particular word,
        # especially if we are using a start token
        logprob = 0
        ct = 0
        for bigram in sentence_ngram_iterator(sentence, 2):
            ct += 1
            # print(ct, bigram)
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
    arg_parser = ArgumentParser("")
    arg_parser.add_argument("--config-file-loc", required=True,
                            type=str,
                            help="absolute path to a JSON file specifying values of this script's config params",
                            metavar="CONFIG_FILE_LOC")

    command_line_args = arg_parser.parse_args()
    with open(command_line_args.config_file_loc, "r") as file:
        dict_config_params = simplejson.load(file)

    smoothing = dict_config_params["smoothing"]
    dictionary_path = dict_config_params["dictionary path"]
    model_path = dict_config_params["model path"]
    test_sentence = dict_config_params["test sentence"]
    max_vocab_size = dict_config_params["max vocab size"]

    start_token = dict_config_params["start token"] or START
    end_token = dict_config_params["end token"] or END

    model = None
    dictionary = None

    if test_sentence is None:
        test_sentence = TEST_SENTENCE

    if os.path.isfile(model_path):
        model = BrownNgramModel.load(model_path)
        dictionary = model.vocab
        start_token = model.start_token
        end_token = model.end_token
        print("loaded model {}".format(model_path))

    corpus_reader = BrownCorpusReader(dictionary=dictionary, start_token=start_token, end_token=end_token)

    if model is None:
        if not os.path.isfile(dictionary_path):
            dictionary = make_and_save_dictionary(corpus_reader, dictionary_path, max_vocab_size)
        else:
            dictionary = Dictionary.load(dictionary_path)
            print("loaded Dictionary from {}".format(dictionary_path))

            if max_vocab_size and len(dictionary) > max_vocab_size:
                print("Dictionary size too large: {}, should be {} max".format(len(dictionary), max_vocab_size))
                dictionary = make_and_save_dictionary(corpus_reader, dictionary_path, max_vocab_size)

            for token in [start_token, end_token]:
                if token not in dictionary:
                    warnings.warn("{} not in dictionary, adding".format(token))
                    dictionary.add_tokens(token)

        print("vocab size: {}".format(len(dictionary)))
        corpus_reader.dictionary = dictionary

        model = BrownNgramModel(corpus_reader, 2)
        model.save(model_path)
        print("created and saved model {}".format(model_path))

    logprob = model.sentence_log_prob(test_sentence, smoothing=smoothing)
    print("{}: {}".format(logprob, " ".join(test_sentence)))

    while True:
        sent1 = corpus_reader.get_random_sentence()
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

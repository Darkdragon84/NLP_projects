import os
import random
import warnings

import simplejson
from argparse import ArgumentParser

from bigram_model.bigram_models import MarkovNgramModel, NeuralNgramModel
from tools.corpus_readers import BrownCorpusReader
from tools.dictionary import Dictionary, make_and_save_dictionary

START = 'START_TOKEN'
END = 'END_TOKEN'
MODEL_TYPES = {"markov": MarkovNgramModel,
               "neural": NeuralNgramModel}

TEST_SENTENCE = "They are trying to demonstrate some different ways of teaching and learning .".split()


def get_random_sentence_from_vocab(dictionary, n_words):
    sentence = [random.choice(dictionary.tokens) for _ in range(n_words)] + ['.']
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

    dictionary_path = dict_config_params["dictionary path"]
    model_path = dict_config_params["model path"]
    test_sentence = dict_config_params["test sentence"]
    max_vocab_size = dict_config_params["max vocab size"]

    start_token = dict_config_params["start token"] or START
    end_token = dict_config_params["end token"] or END

    model = None
    model_class = None

    dictionary = None

    if test_sentence is None:
        test_sentence = TEST_SENTENCE

    if model_path and os.path.isfile(model_path):
        model = MarkovNgramModel.load(model_path)
        model_class = model.__class__

        dictionary = model.vocab
        start_token = model.start_token
        end_token = model.end_token
        print("loaded model {} of type {}".format(model_path, model_class.__name__))
    else:
        model_type = dict_config_params["model type"]
        if model_type not in MODEL_TYPES:
            raise ValueError("model type {} not recognized".format(model_type))
        model_class = MODEL_TYPES[model_type]

    corpus_reader = BrownCorpusReader(dictionary=dictionary, start_token=start_token, end_token=end_token)

    if model is None:

        if not os.path.isfile(dictionary_path):
            dictionary = make_and_save_dictionary(corpus_reader, dictionary_path, max_vocab_size)
        else:
            dictionary = Dictionary.load(dictionary_path)
            print("loaded Dictionary from {}".format(dictionary_path))

            for token in [start_token, end_token]:
                if token not in dictionary:
                    warnings.warn("{} not in dictionary, adding".format(token))
                    dictionary.add_tokens(token)

        print("vocab size: {}".format(len(dictionary)))
        corpus_reader.dictionary = dictionary

        if model_class == MarkovNgramModel:
            model_parameters = dict_config_params["markov ngram parameters"]

            smoothing = model_parameters['smoothing']
            model = MarkovNgramModel(corpus_reader, 2, smoothing=smoothing)
        elif model_class == NeuralNgramModel:
            model_parameters = dict_config_params["neural ngram parameters"]

            learning_rate = model_parameters['learning rate']
            batch_size = model_parameters['batch size']
            epochs = model_parameters['number of epochs']
            model = NeuralNgramModel(corpus_reader, 2)
            model.train(learning_rate, batch_size, epochs)

        if model_path:
            model.save(model_path)
            print("created and saved model {}".format(model_path))

    logprob = model.sentence_log_prob(test_sentence)
    print("{}: {}".format(logprob, " ".join(test_sentence)))

    while True:
        sent1 = corpus_reader.get_random_sentence()
        sent2 = get_random_sentence_from_vocab(dictionary, random.choice(range(5, 20)))

        logprob1 = model.sentence_log_prob(sent1)
        logprob2 = model.sentence_log_prob(sent2)

        print("{}: {}".format(logprob1, " ".join(sent1)))
        print("{}: {}".format(logprob2, " ".join(sent2)))

        choice = input("continue [Y/n]:")
        if choice == 'n':
            break


if __name__ == '__main__':
    main()

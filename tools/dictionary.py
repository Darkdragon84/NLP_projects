

class Dictionary(object):
    def __init__(self):
        self._tokens = list()
        self._token_to_id = dict()

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._token_to_id
        elif isinstance(item, (list, tuple, set)):
            return all(element in self for element in item)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, token):
        return self._token_to_id[token]

    @property
    def tokens(self):
        return self._tokens

    @property
    def token_to_id(self):
        return self._token_to_id

    def token_from_id(self, token_id):
        if token_id >= len(self):
            raise KeyError("id {} not in Dictionary, max. id: {}".format(token_id, len(self) - 1))
        return self._tokens[token_id]

    def add_tokens(self, tokens):
        if not isinstance(tokens, (list, set, tuple)):
            raise ValueError("tokens must be either a list or set of items")

        new_id = len(self._tokens)

        for token in tokens:
            if token not in self:
                self._tokens.append(token)
                self._token_to_id[token] = new_id
                new_id += 1

    def remove_tokens(self, tokens):
        ids_to_remove = list()
        for token in tokens:
            token_id = self._token_to_id.get(token)
            if token_id:
                ids_to_remove.append(token_id)

        for ind in sorted(ids_to_remove, reverse=True):
            del self._tokens[ind]

        self._token_to_id = {i: token for i, token in enumerate(self._tokens)}

    @classmethod
    def from_tokens(cls, word_iterable):
        dic = cls()
        dic.add_tokens(word_iterable)
        return dic


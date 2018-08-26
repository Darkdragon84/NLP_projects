

class Dictionary(object):
    def __init__(self):
        self._tokens = list()
        self._token_to_id = dict()

    def __contains__(self, token):
        return token in self._token_to_id

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, token_id):
        if token_id >= len(self):
            raise KeyError("id {} not in Dictionary, max. id: {}".format(token_id, len(self) - 1))
        return self._tokens[token_id]

    def token_id(self, token):
        return self._token_to_id[token]

    def add_tokens(self, tokens):
        if not isinstance(tokens, (list, set)):
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
    def from_token_iterable(cls, word_iterable):
        dic = cls()
        dic.add_tokens(word_iterable)
        return dic


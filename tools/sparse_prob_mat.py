from collections import defaultdict


class SparseBigramProbabilityMatrix(object):
    def __init__(self, bigram_counts, smoothing=1.):
        self.smoothing = smoothing
        self._bigram_counts = bigram_counts
        self._unigram_counts = None
        self._size = None
        self._compute_unigram_counts()

    def _compute_unigram_counts(self):
        self._unigram_counts = defaultdict(int)
        tokens = set()
        for (a, b), ct in self._bigram_counts.items():
            tokens.add(a)
            tokens.add(b)
            self._unigram_counts[a] += ct
        self._size = len(tokens)

    def __len__(self):
        return self._size

    def __getitem__(self, items):
        a, b = items
        ugc = self._unigram_counts.get(a, None)
        if ugc is None or b not in self._unigram_counts:
            # if a or b are not in unigram counts, they are also not in vocabulary and we assign zero probability
            return 0
        bgc = self._bigram_counts.get((a, b), 0)
        return (bgc + self.smoothing) / (ugc + self._size * self.smoothing)
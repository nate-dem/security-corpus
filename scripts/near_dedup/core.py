"""Approved lexical comparison; MinHash retrieves, full sets decide."""

import hashlib
from functools import lru_cache
from itertools import islice
import re

from datasketch import MinHash
import numpy as np

# Researcher approved 2026-09-17; see docs/filtering.md.
POLICY = {
    "version": "near-dedup-v1",
    "shingle_tokens": 5,
    "similarity_numerator": 9,
    "similarity_denominator": 10,
    "normalization": "case-preserving Unicode word runs and individual non-whitespace punctuation; whitespace ignored",
    "precedence": "all baseline retained; additions by descending cl100k_base length, content hash, input ordinal",
    "representatives": "direct verified retained representative; no transitive removal",
    # Retrieval parameters, not the acceptance threshold. At ideal MinHash
    # Jaccard .9, 16 bands x 8 rows retrieve with probability about .99988.
    "permutations": 128,
    "bands": 16,
    "rows_per_band": 8,
    "seed": 1729,
}
LEXICAL = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def shingles(text):
    tokens = LEXICAL.findall(text)
    if not tokens:
        raise ValueError("Empty lexical content")
    size = POLICY["shingle_tokens"]
    # U+001F is whitespace for the tokenizer, so it cannot occur inside a token.
    # Joining with it preserves token boundaries without string-key ambiguity.
    return {
        "\x1f".join(tokens[i : i + size]) for i in range(max(1, len(tokens) - size + 1))
    }


@lru_cache(maxsize=1)
def permutations():
    sketch = MinHash(num_perm=POLICY["permutations"], seed=POLICY["seed"])
    return sketch.permutations


def signature(values):
    sketch = MinHash(
        num_perm=POLICY["permutations"],
        seed=POLICY["seed"],
        permutations=permutations(),
    )
    iterator = iter(values)
    while batch := list(islice(iterator, 512)):
        sketch.update_batch([s.encode("utf-8") for s in batch])
    return sketch.hashvalues.astype("<u8").tobytes()


def buckets(value):
    width = POLICY["rows_per_band"] * 8
    if len(value) != POLICY["permutations"] * 8:
        raise ValueError("Invalid MinHash signature")
    return [
        (
            i,
            hashlib.blake2b(
                value[i * width : (i + 1) * width], digest_size=16
            ).digest(),
        )
        for i in range(POLICY["bands"])
    ]


def similarity(left, right):
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection, union


def qualifies(intersection, union):
    # Integer comparison avoids moving the researcher's boundary by rounding.
    return union > 0 and intersection * 10 >= union * 9


def permutation_digest():
    return hashlib.sha256(np.asarray(permutations(), dtype="<u8").tobytes()).hexdigest()

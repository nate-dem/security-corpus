import hashlib

import tiktoken

# Module-level cached encoder for token counting.
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of the UTF-8 encoded content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_token_count(content: str) -> int:
    """Return the token count of content using the cl100k_base tokenizer."""
    return len(_get_encoder().encode(content, disallowed_special=()))


# License constants — one per source or license family.
CC_BY_SA_4_0 = "CC-BY-SA-4.0"
CC_BY_SA_3_0 = "CC-BY-SA-3.0"
CC_BY_SA_2_5 = "CC-BY-SA-2.5"
MIT = "MIT"
PUBLIC_DOMAIN = "Public Domain"
MITRE_TERMS = "MITRE Terms of Use"
CISA_TERMS = "CISA Terms of Use"
DETECTION_RULE_LICENSE_LGPL_2_1 = "LGPL-2.1"
CC_BY_4_0 = "CC-BY-4.0"
REDDIT_TERMS = "Reddit Terms of Service"
FLAWS_CLOUD_PUBLIC = "Public Domain (flaws.cloud)"

# arXiv license constants
ARXIV_PERPETUAL_NON_EXCLUSIVE = "arXiv Perpetual Non-Exclusive License"
CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
CC_BY_NC_ND_4_0 = "CC-BY-NC-ND-4.0"

ARXIV_LICENSE_MAP: dict[str, str] = {
    "http://creativecommons.org/licenses/by/4.0/": CC_BY_4_0,
    "http://creativecommons.org/licenses/by-sa/4.0/": CC_BY_SA_4_0,
    "http://creativecommons.org/licenses/by-nc-sa/4.0/": CC_BY_NC_SA_4_0,
    "http://creativecommons.org/licenses/by-nc-nd/4.0/": CC_BY_NC_ND_4_0,
    "http://creativecommons.org/publicdomain/zero/1.0/": PUBLIC_DOMAIN,
    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/": ARXIV_PERPETUAL_NON_EXCLUSIVE,
}

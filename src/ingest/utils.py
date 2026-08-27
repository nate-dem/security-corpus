import hashlib
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

_ENCODER = None
_TOKENIZER_ASSET = Path(__file__).with_name("assets") / "cl100k_base.tiktoken"
_TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_CL100K_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| "
    r"?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)
_CL100K_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        if not _TOKENIZER_ASSET.is_file():
            raise FileNotFoundError(
                f"Packaged cl100k_base vocabulary is missing: {_TOKENIZER_ASSET}"
            )
        digest = hashlib.sha256(_TOKENIZER_ASSET.read_bytes()).hexdigest()
        if digest != _TOKENIZER_SHA256:
            raise RuntimeError(
                "Packaged cl100k_base vocabulary failed its SHA-256 integrity check"
            )
        mergeable_ranks = load_tiktoken_bpe(
            str(_TOKENIZER_ASSET),
            expected_hash=_TOKENIZER_SHA256,
        )
        _ENCODER = tiktoken.Encoding(
            name="cl100k_base",
            pat_str=_CL100K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=_CL100K_SPECIAL_TOKENS,
        )
    return _ENCODER


def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of the UTF-8 encoded content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_token_count(content: str) -> int:
    """Return the token count of content using the cl100k_base tokenizer."""
    return len(_get_encoder().encode(content, disallowed_special=()))


# License identifiers and source-specific terms.
CC_BY_SA_4_0 = "CC-BY-SA-4.0"
CC_BY_SA_3_0 = "CC-BY-SA-3.0"
CC_BY_SA_2_5 = "CC-BY-SA-2.5"
CC_BY_3_0 = "CC-BY-3.0"
CC_BY_NC_SA_3_0 = "CC-BY-NC-SA-3.0"
MIT = "MIT"
PUBLIC_DOMAIN = "Public Domain"
CC0_1_0 = "CC0-1.0"
NOASSERTION = "NOASSERTION"
NVD_CVE_TERMS = "CVE Terms of Use / NIST public data"
MITRE_TERMS = "MITRE Terms of Use"
CISA_KEV_LICENSE = CC0_1_0
DETECTION_RULE_LICENSE = "DRL-1.1"
CC_BY_4_0 = "CC-BY-4.0"
REDDIT_TERMS = NOASSERTION
FLAWS_CLOUD_TERMS = NOASSERTION

# arXiv license constants
ARXIV_PERPETUAL_NON_EXCLUSIVE = "arXiv Perpetual Non-Exclusive License"
CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
CC_BY_NC_ND_4_0 = "CC-BY-NC-ND-4.0"

ARXIV_LICENSE_MAP: dict[str, str] = {
    "http://creativecommons.org/licenses/by/4.0/": CC_BY_4_0,
    "https://creativecommons.org/licenses/by/4.0/": CC_BY_4_0,
    "http://creativecommons.org/licenses/by-sa/4.0/": CC_BY_SA_4_0,
    "https://creativecommons.org/licenses/by-sa/4.0/": CC_BY_SA_4_0,
    "http://creativecommons.org/licenses/by-nc-sa/4.0/": CC_BY_NC_SA_4_0,
    "https://creativecommons.org/licenses/by-nc-sa/4.0/": CC_BY_NC_SA_4_0,
    "http://creativecommons.org/licenses/by-nc-nd/4.0/": CC_BY_NC_ND_4_0,
    "https://creativecommons.org/licenses/by-nc-nd/4.0/": CC_BY_NC_ND_4_0,
    "http://creativecommons.org/licenses/by/3.0/": CC_BY_3_0,
    "https://creativecommons.org/licenses/by/3.0/": CC_BY_3_0,
    "http://creativecommons.org/licenses/by-nc-sa/3.0/": CC_BY_NC_SA_3_0,
    "https://creativecommons.org/licenses/by-nc-sa/3.0/": CC_BY_NC_SA_3_0,
    "http://creativecommons.org/publicdomain/zero/1.0/": CC0_1_0,
    "https://creativecommons.org/publicdomain/zero/1.0/": CC0_1_0,
    "http://creativecommons.org/publicdomain/": PUBLIC_DOMAIN,
    "https://creativecommons.org/publicdomain/": PUBLIC_DOMAIN,
    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/": ARXIV_PERPETUAL_NON_EXCLUSIVE,
    "https://arxiv.org/licenses/nonexclusive-distrib/1.0/": ARXIV_PERPETUAL_NON_EXCLUSIVE,
}

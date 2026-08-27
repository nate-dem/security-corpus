# Packaged tokenizer data

`cl100k_base.tiktoken` is the immutable OpenAI vocabulary used to compute the
corpus `content_length` field. Its SHA-256 digest is recorded in `ingest.utils`.
Keeping the vocabulary with the package makes token counts reproducible and
prevents ingestion jobs from downloading data at runtime.

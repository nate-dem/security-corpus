import duckdb

con = duckdb.connect()


def inspect_source(source_id, n=2):
    """Show full records for n samples from a source."""
    print(f"\n{'='*80}")
    print(f"SOURCE: {source_id}")
    print(f"{'='*80}")
    
    pattern = f"data/{source_id}/normalized/**/*.parquet"
    
    # First, summary stats
    summary = con.execute(f"""
        SELECT COUNT(*) as n_records,
               ROUND(AVG(content_length), 0) as avg_tokens,
               MIN(content_length) as min_tokens,
               MAX(content_length) as max_tokens,
               SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) as empty_content,
               SUM(CASE WHEN content_hash IS NULL THEN 1 ELSE 0 END) as missing_hash,
               SUM(CASE WHEN license IS NULL THEN 1 ELSE 0 END) as missing_license,
               COUNT(DISTINCT content_hash) as unique_hashes
        FROM read_parquet('{pattern}')
    """).fetchone()
    
    n_records, avg_tok, min_tok, max_tok, empty, no_hash, no_license, unique = summary
    print(f"Records: {n_records}")
    print(f"Tokens: avg={avg_tok}, min={min_tok}, max={max_tok}")
    print(f"Empty content: {empty}")
    print(f"Missing hash: {no_hash}")
    print(f"Missing license: {no_license}")
    print(f"Unique hashes: {unique} ({n_records - unique} duplicates)")
    
    # Sample n random records
    samples = con.execute(f"""
        SELECT * FROM read_parquet('{pattern}')
        USING SAMPLE {n}
    """).fetchdf()
    
    for i, row in samples.iterrows():
        print(f"\n--- Sample {i+1} ---")
        for col in samples.columns:
            val = row[col]
            if col == 'content':
                preview = str(val)[:500]
                print(f"  content (first 500 chars):\n    {preview}")
                if len(str(val)) > 500:
                    print(f"    ... [{len(str(val))} chars total]")
            elif col == 'raw':
                if val is None:
                    print(f"  raw: None")
                else:
                    print(f"  raw: <dict, {len(str(val))} chars>")
            else:
                print(f"  {col}: {val}")


sources = [
    "nvd",
    "cisa-kev",
    "mitre-attack",
    "mitre-cwe",
    "mitre-capec",
    "sigma",
    "stackexchange-infosec",
]

for source in sources:
    try:
        inspect_source(source)
    except Exception as e:
        print(f"ERROR inspecting {source}: {e}")
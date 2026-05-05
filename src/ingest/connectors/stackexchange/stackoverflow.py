"""Three-pass streaming connector for Stack Overflow data dumps.

Designed for data that doesn't fit in memory.
Uses intermediate Parquet files and DuckDB for indexed answer lookups.

Pass 1: Stream Posts.xml, collect question IDs with security-relevant
        tags into a set, persist set to disk.
Pass 2: Stream Posts.xml, write answers whose ParentId is in the
        filtered question set to intermediate Parquet in batches.
Pass 3: Stream Posts.xml, filter to questions in the set, query DuckDB
        for answers from intermediate Parquet, assemble + normalize,
        write output Parquet in batches.
"""

import logging
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import lxml.etree as etree
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.connectors.base import QAThreadData
from ingest.connectors.stackexchange.common import (
    assemble_qa_content,
    detect_code_in_html,
    extract_closure,
    html_to_markdown,
    parse_se_datetime,
    parse_tag_string,
)
from ingest.utils import CC_BY_SA_4_0, compute_content_hash, compute_token_count


logger = logging.getLogger(__name__)

# tags that are directly security-relevant
SECURITY_TAGS: set[str] = {
    'security',
    'authentication',
    'network-programming',
    'encryption',
    'spring-security',
    'oauth',
    'jwt',
    'cryptography',
    'authorization',
    'ssl',
    'https',
    'xss',
    'sql-injection',
    'csrf',
    'hash',
    'x509',
    'pki',
    'ssh',
    'firewall',
    'cors',
    'penetration-testing',
    'oauth-2.0',
    'openid-connect',
    'saml',
    'password-encryption',
    'password-hash',
    'digital-signature',
    'privilege',
    'openssl',
    'aes',
    'rsa',
    'md5',
    'sha256',
    'sha1',
    'bcrypt',
    'hmac',
    'public-key-encryption',
    'private-key',
    'public-key',
    'salt-cryptography',
    'encryption-symmetric',
    'encryption-asymmetric',
    'cryptojs',
    'pycrypto',
    'ssl-certificate',
    'certificate',
    'x509certificate',
    'tls1.2',
    'lets-encrypt',
    'client-certificates',
    'self-signed',
    'keystore',
    'keytool',
    'single-sign-on',
    'keycloak',
    'ldap',
    'access-token',
    'bearer-token',
    'kerberos',
    'basic-authentication',
    'saml-2.0',
    'auth0',
    'content-security-policy',
    'owasp',
    'csrf-protection',
    'code-injection',
    'session-cookies',
    'passwords',
    'password-protection',
    'dns',
    'vpn',
    'proxy',
    'reverse-proxy',
    'wireshark',
    'linux',
    'http',
    'kernel',
    'ip',
    'thread-safety',
    'segmentation-fault',
    'gdb',
    'memory-leaks',
    'google-oauth',
    'amazon-s3',
    'server',
    'ubuntu',
    'operating-system',
    'tcp',
    'udp',
    'tls',
    'iptables',
    'nmap',
    'packet',
    'pcap',
    'scapy',
    'sockets',
    'network-security',
    'web-security',
    'buffer-overflow',
    'reverse-engineering',
    'malware',
    'exploit',
    'vulnerability',
    'injection',
    'sanitization',
    'input-validation',
    'brute-force',
    'active-directory',
    'azure-active-directory',
    'access-control',
    'role-based-access-control',
    'permissions',
    'session',
    'cookie',
    'tls1.3',
    'gnupg',
    'gpg',
    'bouncycastle',
    'libsodium',
    'diffie-hellman',
    'elliptic-curve',
    'ecdsa',
    'aws-iam',
    'aws-security',
    'selinux',
    'apparmor',
    'seccomp',
    'sandbox',
}


def _iter_post_elements(source):
    """Yield attr dicts for each <row> element from a Posts.xml source.

    source can be a Path to an XML file (for testing) or a file-like
    object such as subprocess.stdout for actual ingestion.
    """
    src = str(source) if isinstance(source, Path) else source
    for _, elem in etree.iterparse(src, events=("end",), tag="row"):
        attrs = dict(elem.attrib)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        yield attrs


@contextmanager
def _open_posts_stream(archive_path: Path):
    """Context manager that streams Posts.xml from a 7z archive via subprocess.

    Yields an iterable of attr dicts. Each call re-decompresses the archive.
    """
    proc = subprocess.Popen(
        ["7z", "x", "-so", str(archive_path), "Posts.xml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield _iter_post_elements(proc.stdout)
    finally:
        proc.terminate()
        proc.wait()


class StackOverflowConnector:
    """Three-pass streaming connector for Stack Overflow data dumps.

    Designed for data that doesn't fit in memory (~100GB Posts.xml).
    Uses intermediate Parquet files and DuckDB for indexed answer lookups.

    Pass 1: Stream Posts.xml, collect question IDs with security-relevant
            tags into a set, persist set to disk.
    Pass 2: Stream Posts.xml, write answers whose ParentId is in the
            filtered question set to intermediate Parquet in batches.
    Pass 3: Stream Posts.xml, filter to questions in the set, query DuckDB
            for answers from intermediate Parquet, assemble + normalize,
            write output Parquet in batches.
    """

    source_id = "stackoverflow"

    def ingest(
        self,
        archive_path: Path,
        output_dir: Path,
        intermediate_dir: Path,
        batch_size: int = 100_000,
    ) -> int:
        """Run the three-pass ingestion pipeline. Returns record count."""

        question_ids = self._collect_question_ids(archive_path, intermediate_dir)
        self._write_answer_index(
            archive_path, question_ids, intermediate_dir, batch_size
        )
        count = self._assemble_and_write(
            archive_path, question_ids, intermediate_dir, output_dir, batch_size
        )
        return count

    def _collect_question_ids(
        self, archive_path: Path, intermediate_dir: Path
    ) -> set[int]:
        """Pass 1: Collect question IDs with security-relevant tags."""
        marker = intermediate_dir / "_question_ids.DONE"
        ids_path = intermediate_dir / "question_ids.parquet"

        if marker.exists():
            logger.info(
                "Pass 1/3: Loading cached question IDs from %s", ids_path
            )
            table = pq.read_table(ids_path)
            return set(table.column("question_id").to_pylist())

        logger.info("Pass 1/3: Collecting security-tagged question IDs...")
        question_ids: set[int] = set()
        scanned = 0

        with _open_posts_stream(archive_path) as posts:
            for attrs in posts:
                scanned += 1
                if scanned % 1_000_000 == 0:
                    logger.info(
                        "  Pass 1: scanned %dM posts, %d questions matched",
                        scanned // 1_000_000,
                        len(question_ids),
                    )

                if attrs.get("PostTypeId") != "1":
                    continue

                tags = parse_tag_string(attrs.get("Tags", ""))
                if set(tags) & SECURITY_TAGS:
                    question_ids.add(int(attrs["Id"]))

        # Persist to disk for resumability
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({"question_id": list(question_ids)})
        pq.write_table(table, ids_path, compression="snappy")
        marker.touch()

        logger.info(
            "Pass 1/3: Found %d security-tagged questions", len(question_ids)
        )
        return question_ids

    def _write_answer_index(
        self,
        archive_path: Path,
        question_ids: set[int],
        intermediate_dir: Path,
        batch_size: int,
    ) -> int:
        """Pass 2: Write answers for security-tagged questions to intermediate Parquet."""
        answers_dir = intermediate_dir / "answers"
        marker = answers_dir / "_DONE"

        if marker.exists():
            logger.info("Pass 2/3: Answer index already built, skipping")
            total = 0
            for f in answers_dir.glob("batch_*.parquet"):
                total += pq.read_metadata(f).num_rows
            return total

        # Incomplete previous run — clean up and restart
        if answers_dir.exists():
            logger.warning(
                "Pass 2/3: Incomplete answer index found, restarting"
            )
            shutil.rmtree(answers_dir)

        logger.info("Pass 2/3: Writing answer index to intermediate Parquet...")
        answers_dir.mkdir(parents=True, exist_ok=True)

        batch: list[dict] = []
        batch_num = 0
        total_answers = 0
        scanned = 0

        with _open_posts_stream(archive_path) as posts:
            for attrs in posts:
                scanned += 1
                if scanned % 1_000_000 == 0:
                    logger.info(
                        "  Pass 2: scanned %dM posts, %d answers matched",
                        scanned // 1_000_000,
                        total_answers,
                    )

                if attrs.get("PostTypeId") != "2":
                    continue

                parent_id = attrs.get("ParentId")
                if parent_id is None:
                    continue
                if int(parent_id) not in question_ids:
                    continue

                batch.append(
                    {
                        "id": int(attrs["Id"]),
                        "parent_id": int(parent_id),
                        "body_html": attrs.get("Body", ""),
                        "score": int(attrs.get("Score", 0)),
                    }
                )
                total_answers += 1

                if len(batch) >= batch_size:
                    _flush_answer_batch(batch, answers_dir, batch_num)
                    batch_num += 1
                    batch = []

        if batch:
            _flush_answer_batch(batch, answers_dir, batch_num)
            batch_num += 1

        marker.touch()
        logger.info(
            "Pass 2/3: Wrote %d answers in %d batches",
            total_answers,
            batch_num,
        )
        return total_answers

    def _assemble_and_write(
        self,
        archive_path: Path,
        question_ids: set[int],
        intermediate_dir: Path,
        output_dir: Path,
        batch_size: int,
    ) -> int:
        """Pass 3: Assemble Q&A documents and write output Parquet."""
        logger.info("Pass 3/3: Assembling Q&A documents and writing output...")

        answers_dir = intermediate_dir / "answers"
        has_answer_index = (
            answers_dir.exists()
            and any(answers_dir.glob("batch_*.parquet"))
        )

        con = duckdb.connect()
        if has_answer_index:
            answers_glob = str(answers_dir / "batch_*.parquet")
            con.execute(
                f"CREATE VIEW answers AS "
                f"SELECT * FROM read_parquet('{answers_glob}')"
            )

        batch: list[QAThreadData] = []
        batch_num = 0
        total_records = 0

        with _open_posts_stream(archive_path) as posts:
            for attrs in posts:
                if attrs.get("PostTypeId") != "1":
                    continue

                qid = int(attrs["Id"])
                if qid not in question_ids:
                    continue

                # Query answers from DuckDB
                if has_answer_index:
                    answer_rows = con.execute(
                        "SELECT id, body_html, score FROM answers "
                        "WHERE parent_id = ? ORDER BY score DESC",
                        [qid],
                    ).fetchall()
                else:
                    answer_rows = []

                # Build question dict (same fields as site.py)
                body_html = attrs.get("Body", "")
                accepted_answer_id = (
                    int(attrs["AcceptedAnswerId"])
                    if attrs.get("AcceptedAnswerId")
                    else None
                )

                question = {
                    "id": qid,
                    "title": attrs.get("Title", ""),
                    "body_html": body_html,
                    "body_md": html_to_markdown(body_html),
                    "creation_date": attrs.get("CreationDate"),
                    "score": int(attrs.get("Score", 0)),
                    "answer_count": int(attrs.get("AnswerCount", 0)),
                    "accepted_answer_id": accepted_answer_id,
                    "tags": parse_tag_string(attrs.get("Tags", "")),
                    "has_code": detect_code_in_html(body_html),
                }

                # Build assembled answers list
                answers = []
                for row in answer_rows:
                    ans_id, ans_html, ans_score = row
                    answers.append(
                        {
                            "id": ans_id,
                            "body_html": ans_html,
                            "body_md": html_to_markdown(ans_html),
                            "score": ans_score,
                            "is_accepted": (
                                ans_id == accepted_answer_id
                                if accepted_answer_id
                                else False
                            ),
                            "has_code": detect_code_in_html(ans_html),
                        }
                    )

                closed = extract_closure(attrs)
                content = assemble_qa_content(question, answers)

                record = QAThreadData(
                    record_id=f"stackoverflow:question-{qid}",
                    source_id=self.source_id,
                    source_record_id=f"question-{qid}",
                    content=content,
                    content_length=compute_token_count(content),
                    content_hash=compute_content_hash(content),
                    title=question["title"],
                    ingested_at=datetime.now(timezone.utc),
                    license=CC_BY_SA_4_0,
                    published_at=parse_se_datetime(question.get("creation_date")),
                    source_url=f"https://stackoverflow.com/questions/{qid}",
                    score=question["score"],
                    answer_count=len(answers),
                    has_accepted_answer=accepted_answer_id is not None,
                    closed=closed,
                    tags=question["tags"],
                    raw=None,
                )

                batch.append(record)
                total_records += 1

                if len(batch) >= batch_size:
                    _flush_output_batch(batch, output_dir, batch_num)
                    batch_num += 1
                    batch = []

        if batch:
            _flush_output_batch(batch, output_dir, batch_num)
            batch_num += 1

        con.close()
        logger.info("Pass 3/3: Wrote %d records", total_records)
        return total_records


def _flush_answer_batch(
    batch: list[dict], answers_dir: Path, batch_num: int
):
    """Write a batch of answer records to intermediate Parquet."""
    table = pa.table(
        {
            "id": pa.array([r["id"] for r in batch], type=pa.int64()),
            "parent_id": pa.array(
                [r["parent_id"] for r in batch], type=pa.int64()
            ),
            "body_html": pa.array(
                [r["body_html"] for r in batch], type=pa.string()
            ),
            "score": pa.array([r["score"] for r in batch], type=pa.int64()),
        }
    )
    out = answers_dir / f"batch_{batch_num:06d}.parquet"
    pq.write_table(table, out, compression="snappy")


def _flush_output_batch(
    batch: list[QAThreadData], output_dir: Path, batch_num: int
):
    """Write a batch of QAThreadData records to output Parquet."""
    rows = [r.model_dump() for r in batch]
    table = pa.Table.from_pylist(rows)
    out = output_dir / "source_id=stackoverflow" / f"batch_{batch_num:06d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy")

"""Upload a prepared package, refusing to send data to a public repository."""

import argparse
import hashlib
from pathlib import Path
import re


RELEASE_FILES = {"README.md", ".gitattributes", "SHA256SUMS", "technical_report.pdf"}
FILTERING_FILES = {
    f"metadata/filtering/{name}" for name in (
        "README.md", "selection-policy.json", "web-records.parquet",
        "youtube-records.parquet", "youtube-spans.parquet",
    )
}


def allowed_release_path(relative):
    return (relative in RELEASE_FILES | FILTERING_FILES
            or re.fullmatch(r"data/\d{3}/\d{6}\.parquet", relative) is not None)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ignored(relative):
    return relative.startswith(".cache/") or Path(relative).name == ".DS_Store"


def verify_package(folder):
    folder = Path(folder).resolve()
    expected = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = folder / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.resolve().is_relative_to(folder)):
            raise ValueError(f"Unsafe package path: {relative}")
        if relative in expected:
            raise ValueError(f"Repeated checksum entry: {relative}")
        if not allowed_release_path(relative):
            raise ValueError(f"File is outside the CyberOpen release contents: {relative}")
        if sha(path) != digest:
            raise ValueError(f"Package checksum failed: {relative}")
        expected[relative] = digest
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*")
              if p.is_file() and not ignored(str(p.relative_to(folder)))}
    if actual != set(expected) | {"SHA256SUMS"}:
        raise ValueError("Package contains unlisted or missing files")
    if not {"README.md", ".gitattributes"}.issubset(expected):
        raise ValueError("Missing dataset card or file configuration")
    training = [dict(package_path=p, sha256=digest) for p, digest in sorted(expected.items())
                if p.startswith("data/")]
    if not training:
        raise ValueError("Package contains no training files")
    return dict(training_files=training, files=expected)


def upload_private(folder, repo_id, api):
    inventory = verify_package(folder)
    print(f"Verified {len(inventory['training_files']):,} training files before upload.", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    if api.dataset_info(repo_id=repo_id).private is not True:
        raise ValueError("Repository is public. No files uploaded. Choose a new private repository.")
    # Do not leave old development artifacts in an existing destination.
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    extras = remote_files - set(inventory["files"]) - {"SHA256SUMS"}
    if extras:
        raise ValueError("Destination contains files outside this release. Use a new empty private repository.")
    result = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(Path(folder).resolve()),
        path_in_repo="", ignore_patterns=[".cache/**", ".DS_Store", "**/.DS_Store"],
        commit_message="Upload CyberOpen",
    )
    if api.dataset_info(repo_id=repo_id).private is not True:
        raise RuntimeError("Upload completed, but repository visibility changed; check its settings.")
    print(f"Private dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"Upload result: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="YOUR_ORG/CyberOpen")
    parser.add_argument("--folder", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/huggingface/CyberOpen")
    args = parser.parse_args()
    if args.repo_id.count("/") != 1 or any(v.startswith("YOUR_") for v in args.repo_id.split("/")):
        parser.error("Replace YOUR_ORG with your actual Hugging Face organization or username")
    from huggingface_hub import HfApi
    upload_private(args.folder, args.repo_id, HfApi())


if __name__ == "__main__":
    main()

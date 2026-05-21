"""Clone the GitHub Advisory Database into data/github-advisory/raw/.

Usage:
    python scripts/export/github_advisory.py

The script does a shallow clone (--depth=1) so it only fetches the latest
snapshot without full git history (~700 MB checked-out, ~200 MB transfer).
If the target directory already exists, it does a git pull instead so
subsequent runs are incremental.
"""
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/github/advisory-database.git"
RAW_DIR = Path("data/github-advisory/raw")


def main() -> None:
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / ".git").is_dir():
        print(f"Updating existing clone at {RAW_DIR} …")
        result = subprocess.run(
            ["git", "-C", str(RAW_DIR), "pull", "--depth=1", "--ff-only"],
            check=False,
        )
    else:
        print(f"Cloning {REPO_URL} into {RAW_DIR} …")
        result = subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(RAW_DIR)],
            check=False,
        )

    if result.returncode != 0:
        print("git command failed — see output above.", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()

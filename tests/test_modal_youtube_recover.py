import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts import modal_youtube_recover as recovery
from scripts.curation import runtime
from scripts.curation_single import run
from scripts.modal_youtube import common as c, transfer
from test_modal_youtube import (
    handoff as _handoff,
    single_env as _single_env,
    production_env as _production_env,
)

handoff, single_env, production_env = _handoff, _single_env, _production_env


class Volume:
    def commit(self):
        pass


def test_recovery_without_coordinator_or_worker_inventory_preserves_and_imports(
    handoff,
    tmp_path_factory,
):
    path, plan, folder, loads, calls = handoff
    manifest_path = folder / "manifest.json"
    manifest = c.read(manifest_path)
    digest = c.sha(manifest_path)
    slot = manifest["jobs"][0][0]
    run.run_slot(path, plan, slot, runtime.ResidentModel())
    directory = run.folder(plan, slot)
    c.write(directory / "modal-origin.json", {"manifest_sha256": digest})
    original = {p: c.sha(p) for p in c.part_files(directory)}
    root = Path(manifest["mount"])
    remote_manifest = root / "modal-handoff/manifest.json"
    remote_manifest.parent.mkdir()
    remote_manifest.write_bytes(manifest_path.read_bytes())
    ledger = {"launch": {"manifest_sha256": digest}}
    archive_root = tmp_path_factory.mktemp("recovery-volume")
    recovery.build(root, digest, Volume(), ledger, output_root=archive_root)
    assert original == {p: c.sha(p) for p in c.part_files(directory)}
    assert ledger == {"launch": {"manifest_sha256": digest}}
    assert not (root / recovery.PREFIX).exists()
    catalog = c.read(archive_root / "catalog.json")
    assert catalog["complete"]
    assert sum(v["completed_parts"] for v in catalog["workers"].values()) == 1
    recovery.build(root, digest, Volume(), ledger, output_root=archive_root)
    returned = folder / "returned"
    output = Path(plan["output_dir"]).relative_to(root)
    for worker, group in enumerate(manifest["jobs"]):
        inventory = c.read(archive_root / f"worker-{worker}.json")
        allowed = [output / "parts" / f"{s:05d}" for s in group]
        recovery.extract_checked(
            archive_root / f"worker-{worker}.tar.gz",
            returned,
            inventory,
            allowed,
        )
        c.write(returned / f"worker-{worker}.json", inventory)
    c.write(
        returned / "downloaded.json",
        {"manifest_sha256": digest, "workers": list(range(len(manifest["jobs"])))},
    )
    shutil.rmtree(directory)  # Simulate the unfinished original Marlowe location.
    before = len(loads), list(calls)
    transfer.merge(manifest_path)
    assert c.read(folder / "import-summary.json")["imported_slots"] == [slot]
    assert (len(loads), calls) == before
    run.audit_part(path, plan, slot)


def archive_fixture(
    tmp_path, name="single/parts/00000/file", data=b"data", kind=tarfile.REGTYPE
):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as f:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.size = len(data) if kind == tarfile.REGTYPE else 0
        member.linkname = "outside"
        f.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    inventory = {
        "files": {
            name: {
                "bytes": len(data),
                "sha256": recovery.hashlib.sha256(data).hexdigest(),
            }
        }
    }
    return archive, inventory


@pytest.mark.parametrize("name", ["../outside", "/absolute", "single/parts/00001/file"])
def test_archive_cannot_escape_allocated_partition(tmp_path, name):
    archive, inventory = archive_fixture(tmp_path, name)
    with pytest.raises(ValueError):
        recovery.extract_checked(
            archive, tmp_path / "returned", inventory, [Path("single/parts/00000")]
        )


def test_archive_links_and_changed_content_fail(tmp_path):
    archive, inventory = archive_fixture(tmp_path, kind=tarfile.SYMTYPE)
    with pytest.raises(ValueError, match="archive member"):
        recovery.extract_checked(
            archive, tmp_path / "returned", inventory, [Path("single/parts/00000")]
        )
    archive, inventory = archive_fixture(tmp_path)
    next(iter(inventory["files"].values()))["sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        recovery.extract_checked(
            archive, tmp_path / "returned", inventory, [Path("single/parts/00000")]
        )


def test_incomplete_inventory_and_conflicting_existing_outputs_fail(tmp_path):
    archive, inventory = archive_fixture(tmp_path)
    target = tmp_path / "returned"
    recovery.extract_checked(archive, target, inventory, [Path("single/parts/00000")])
    recovery.extract_checked(archive, target, inventory, [Path("single/parts/00000")])
    (target / "single/parts/00000/file").write_text("changed")
    with pytest.raises(ValueError, match="Conflicting"):
        recovery.extract_checked(
            archive, target, inventory, [Path("single/parts/00000")]
        )
    missing = json.loads(json.dumps(inventory))
    missing["files"]["single/parts/00000/missing"] = inventory["files"][
        "single/parts/00000/file"
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        recovery.extract_checked(
            archive, tmp_path / "other", missing, [Path("single/parts/00000")]
        )

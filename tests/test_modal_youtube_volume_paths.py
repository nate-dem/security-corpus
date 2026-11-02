from pathlib import Path

import pytest

from scripts.curation import runtime
from scripts.curation_single import run
from scripts.modal_youtube import common as c
from test_modal_youtube import (
    handoff as _handoff,
    single_env as _single_env,
    production_env as _production_env,
)

handoff, single_env, production_env = _handoff, _single_env, _production_env


@pytest.fixture
def mounted(handoff):
    path, plan, directory, loads, calls = handoff
    manifest = c.read(directory / "manifest.json")
    mount = Path(manifest["mount"])
    physical = mount.with_name(mount.name + "-physical-volume")
    mount.rename(physical)
    mount.symlink_to(physical, target_is_directory=True)
    try:
        yield path, plan, manifest, loads, calls
    finally:
        mount.unlink()
        physical.rename(mount)


def test_real_scorer_plan_and_checkpoint_resume_through_volume_alias(mounted):
    path, plan, manifest, loads, calls = mounted
    before = path.read_bytes()
    with pytest.raises(ValueError, match="output binding"):
        run.load_plan(path)  # Reproduce the reported Modal failure.
    assert c.load_scoring_plan(manifest, path) == plan
    assert path.resolve() != path  # No global pathlib patch.
    inference = len(loads), list(calls)
    assert run.run_slot(path, plan, 0, runtime.ResidentModel()) == 0
    assert (len(loads), calls) == inference  # Existing work remains cached.
    assert run.run_slot(path, plan, 1, runtime.ResidentModel()) == 0
    counts, _ = run.audit_part(path, plan, 1)
    assert counts["candidate_records"] == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["code", "version", "output", "input"])
def test_alias_does_not_disable_frozen_scorer_checks(mounted, change):
    path, plan, manifest, _, _ = mounted
    if change == "code":
        plan["code"] = {}
    elif change == "version":
        plan["version"] = "wrong"
    elif change == "output":
        plan["output_dir"] = str(path.parent / "wrong")
    else:
        plan["manifest_sha256"] = "wrong"
    c.write(path, plan)
    # Even a correspondingly rehashed handoff cannot disable scorer validation.
    manifest["plan_sha256"] = c.sha(path)
    with pytest.raises(ValueError):
        c.load_scoring_plan(manifest, path)


def test_alias_rejects_plan_checksum_change(mounted):
    path, _, manifest, _, _ = mounted
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="plan changed"):
        c.load_scoring_plan(manifest, path)


def test_alias_rejects_nested_symlink_even_with_identical_file(mounted):
    path, _, manifest, _, _ = mounted
    copied = path.with_name("other-plan.json")
    path.rename(copied)
    path.symlink_to(copied.name)
    with pytest.raises(ValueError, match="Unexpected symlink"):
        c.load_scoring_plan(manifest, path)


def test_non_volume_paths_keep_working(handoff):
    path, plan, directory, _, _ = handoff
    manifest = c.read(directory / "manifest.json")
    assert c.load_scoring_plan(manifest, path) == plan
    with pytest.raises(ValueError, match="mount binding"):
        c.load_scoring_plan(manifest, path.with_name("wrong.json"))

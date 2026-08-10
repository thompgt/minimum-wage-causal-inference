import json

import pytest

from src.data.manifest import (
    build_manifest,
    format_manifest,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def raw_inputs(tmp_path):
    """One present input and one absent one, the two cases worth recording."""
    present = tmp_path / "bls_unemployment.parquet"
    present.write_bytes(b"pretend this is a parquet")
    return {
        "bls_unemployment": present,
        "user_minimum_wage_csv": tmp_path / "state_minimum_wage.csv",
    }


@pytest.fixture
def manifest_path(tmp_path):
    return tmp_path / "manifest.json"


def test_hash_matches_hashlib(raw_inputs):
    import hashlib
    path = raw_inputs["bls_unemployment"]
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_absent_inputs_are_recorded_not_skipped(raw_inputs):
    """A manifest has to say what was *not* there, or substitution is silent."""
    inputs = build_manifest(sources=raw_inputs)["inputs"]
    assert inputs["bls_unemployment"]["present"]
    assert inputs["user_minimum_wage_csv"] == {
        "present": False,
        "path": inputs["user_minimum_wage_csv"]["path"],
    }


def test_write_then_verify_is_ok(raw_inputs, manifest_path):
    write_manifest(path=manifest_path, sources=raw_inputs)
    checked = verify_manifest(path=manifest_path, sources=raw_inputs)
    assert {c["status"] for c in checked.values()} == {"ok"}


def test_a_changed_input_is_detected(raw_inputs, manifest_path):
    write_manifest(path=manifest_path, sources=raw_inputs)
    raw_inputs["bls_unemployment"].write_bytes(b"a different vintage entirely")
    checked = verify_manifest(path=manifest_path, sources=raw_inputs)
    assert checked["bls_unemployment"]["status"] == "changed"
    assert checked["bls_unemployment"]["recorded_sha256"] != (
        checked["bls_unemployment"]["sha256"]
    )


def test_a_substituted_input_is_detected(raw_inputs, manifest_path):
    """The user CSV silently takes precedence over the download; catch it."""
    write_manifest(path=manifest_path, sources=raw_inputs)
    raw_inputs["user_minimum_wage_csv"].write_text("state,year,month,minimum_wage\n")
    checked = verify_manifest(path=manifest_path, sources=raw_inputs)
    assert checked["user_minimum_wage_csv"]["status"] == "unexpected"


def test_a_deleted_input_is_detected(raw_inputs, manifest_path):
    write_manifest(path=manifest_path, sources=raw_inputs)
    raw_inputs["bls_unemployment"].unlink()
    checked = verify_manifest(path=manifest_path, sources=raw_inputs)
    assert checked["bls_unemployment"]["status"] == "missing"


def test_the_second_fetcher_does_not_erase_the_first(raw_inputs, manifest_path):
    """fetch_bls and fetch_minwage write separately and both notes must survive."""
    write_manifest(path=manifest_path, sources=raw_inputs,
                   extra={"bls": {"api_version": 1}})
    write_manifest(path=manifest_path, sources=raw_inputs,
                   extra={"minimum_wage": {"source": "Vaghul-Zipperer"}})
    recorded = json.loads(manifest_path.read_text())
    assert recorded["bls"]["api_version"] == 1
    assert recorded["minimum_wage"]["source"] == "Vaghul-Zipperer"


def test_corrupt_manifest_is_overwritten_rather_than_crashing(raw_inputs, manifest_path):
    manifest_path.write_text("{ not json")
    write_manifest(path=manifest_path, sources=raw_inputs)
    assert load_manifest(manifest_path)["inputs"]["bls_unemployment"]["present"]


def test_no_manifest_reads_as_no_manifest(manifest_path):
    assert load_manifest(manifest_path) is None
    assert verify_manifest(path=manifest_path) is None
    assert "No data manifest" in format_manifest(load_manifest(manifest_path))


def test_format_mentions_the_api_version_and_the_source(raw_inputs, manifest_path):
    manifest = write_manifest(
        path=manifest_path, sources=raw_inputs,
        extra={
            "bls": {"api_version": 2, "start_year": 2000, "end_year": 2022,
                    "measures": ["unemployment_rate", "labor_force"]},
            "minimum_wage": {"source": "Vaghul-Zipperer v1.4.0"},
        },
    )
    text = format_manifest(manifest)
    assert "API v2" in text
    assert "2000-2022" in text
    assert "labor_force" in text
    assert "Vaghul-Zipperer v1.4.0" in text
    assert "user_minimum_wage_csv: absent" in text

"""Record which vintage of the raw inputs produced a given set of results.

`data/` is gitignored, which is right — a 14,000-row API pull is not
source code — but it means a reader of this repository cannot tell which
BLS revision or which release of the minimum wage series the numbers in
the README came from. LAUS is revised: the seasonal adjustment is redone
every January and the model-based state estimates are re-benchmarked, so
the same fetch code run a year apart returns different unemployment rates
for the same state-months. Without a vintage record, "I re-ran it and got
something else" is unresolvable.

`data/manifest.json` is the record, and it is **committed** — it is a few
hundred bytes of provenance, not data. For each raw input it stores the
size, the SHA-256, and the modification time; for the BLS pull it also
stores the API version and the year range requested. `python -m
src.data.manifest` prints it, `verify_manifest()` says whether the files
on disk are still the ones described, and the notebooks print it at the
top so an executed notebook carries its own provenance.
"""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

#: Raw inputs worth pinning. Missing ones are recorded as absent rather
#: than skipped, so a manifest says what *was not* there too.
TRACKED_INPUTS = {
    "bls_unemployment": RAW_DIR / "bls_unemployment.parquet",
    "vz_minimum_wage": RAW_DIR / "vz_v1.4.0_mw_state_monthly.xlsx",
    "user_minimum_wage_csv": RAW_DIR / "state_minimum_wage.csv",
}

_CHUNK = 1 << 20
_UNSET = object()


def _display_path(path):
    """Repo-relative and forward-slashed, so the manifest diffs across OSes.

    Absolute for anything outside the repo (only tests do that).
    """
    path = Path(path)
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path, chunk_size=_CHUNK):
    """Streaming SHA-256, so a large parquet does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path):
    """Size, hash and mtime for one raw input, or `present: False`."""
    path = Path(path)
    if not path.exists():
        return {"present": False, "path": _display_path(path)}
    stat = path.stat()
    return {
        "present": True,
        "path": _display_path(path),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=UTC
        ).isoformat(timespec="seconds"),
    }


def build_manifest(sources=None, extra=None):
    """Describe every tracked raw input as of now.

    `extra` is merged in at the top level — `fetch_bls` uses it to record
    the API version and year range, which are not recoverable from the
    parquet.
    """
    sources = TRACKED_INPUTS if sources is None else sources
    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {name: describe_file(path) for name, path in sources.items()},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path=None, sources=None, extra=None, merge=True):
    """Write the manifest, preserving top-level keys from earlier writes.

    The two fetchers run separately, and each knows things the other does
    not. With `merge` the second to run does not erase the first's notes.
    """
    path = Path(path) if path else MANIFEST_PATH
    manifest = build_manifest(sources=sources, extra=extra)
    if merge and path.exists():
        try:
            previous = json.loads(path.read_text())
        except json.JSONDecodeError:
            previous = {}
        previous.pop("inputs", None)
        previous.pop("generated_utc", None)
        manifest = {**previous, **manifest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(path=None):
    """The recorded manifest, or None if nothing has been fetched here."""
    path = Path(path) if path else MANIFEST_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def verify_manifest(path=None, sources=None):
    """Compare the files on disk against the manifest.

    Returns a dict per input with `status` in {ok, changed, missing,
    unexpected, untracked}. `unexpected` means the manifest recorded the
    file as absent but it is here now — a silently substituted input,
    which is the case worth catching.
    """
    recorded = load_manifest(path)
    if recorded is None:
        return None
    sources = TRACKED_INPUTS if sources is None else sources
    out = {}
    for name, file_path in sources.items():
        current = describe_file(file_path)
        was = recorded.get("inputs", {}).get(name)
        if was is None:
            out[name] = {"status": "untracked", **current}
        elif not was.get("present") and not current["present"]:
            out[name] = {"status": "ok", **current}
        elif not was.get("present"):
            out[name] = {"status": "unexpected", **current}
        elif not current["present"]:
            out[name] = {"status": "missing", **current}
        else:
            same = current["sha256"] == was.get("sha256")
            out[name] = {"status": "ok" if same else "changed",
                         "recorded_sha256": was.get("sha256"), **current}
    return out


def format_manifest(manifest=_UNSET):
    """A few lines fit to print at the top of a notebook."""
    manifest = load_manifest() if manifest is _UNSET else manifest
    if manifest is None:
        return (
            "No data manifest: nothing has been fetched into data/raw/ here, "
            "so any results come from the seeded synthetic panel."
        )
    lines = [f"Data vintage recorded {manifest['generated_utc']}"]
    if "bls" in manifest:
        bls = manifest["bls"]
        lines.append(
            f"  BLS LAUS API v{bls.get('api_version')}, "
            f"{bls.get('start_year')}-{bls.get('end_year')}, "
            f"measures {', '.join(bls.get('measures', []))}"
        )
    if "minimum_wage" in manifest:
        mw = manifest["minimum_wage"]
        lines.append(f"  Minimum wage source: {mw.get('source')}")
    for name, info in sorted(manifest.get("inputs", {}).items()):
        if not info.get("present"):
            lines.append(f"  {name}: absent")
            continue
        lines.append(
            f"  {name}: {info['sha256'][:12]}... "
            f"({info['bytes']:,} bytes, modified {info['modified_utc']})"
        )
    return "\n".join(lines)


def main():
    print(format_manifest())
    checked = verify_manifest()
    if checked is None:
        return
    drifted = {n: c["status"] for n, c in checked.items() if c["status"] != "ok"}
    if drifted:
        print("\nFiles no longer match the manifest:")
        for name, status in sorted(drifted.items()):
            print(f"  {name}: {status}")
        print("Re-run the fetchers to re-record, or treat results as unpinned.")
    else:
        print("\nAll tracked inputs match the manifest.")


if __name__ == "__main__":
    main()

"""Fetch the opt-in public validation corpus; preserve provenance and verify hashes.

Network never runs during ordinary pytest. Use --lock only when deliberately
updating sources; the checked-in manifest otherwise requires exact SHA-256 matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def extract(path: Path, root: Path, limit: int, *, verify_only: bool = False):
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > limit:
            raise ValueError(f"Expanded archive exceeds {limit} bytes: {path.name}")
        for info in archive.infolist():
            output = (root / info.filename).resolve()
            if not output.is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not allowed.")
        if verify_only:
            for info in archive.infolist():
                if not info.is_dir():
                    output = root / info.filename
                    if (
                        not output.is_file()
                        or digest(output) != hashlib.sha256(archive.read(info)).hexdigest()
                    ):
                        raise ValueError(f"Missing or modified unpacked fixture: {output}")
        else:
            archive.extractall(root)


def fetch(entry: dict, root: Path, *, lock: bool, verify_only: bool, limit: int) -> dict:
    target = (root / entry["path"]).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Fixture path escapes the cache.")
    expected = entry.get("sha256")
    if not expected and not lock:
        raise ValueError(f"{entry['id']}: missing locked checksum")
    if target.is_file() and (not expected or digest(target) == expected):
        actual = digest(target)
    else:
        if verify_only:
            raise ValueError(f"{entry['id']}: missing or modified fixture; rerun the downloader")
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "ray360-validation/0.1"}
        )
        temporary = None
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if int(response.headers.get("Content-Length", 0)) > limit:
                    raise ValueError(f"{entry['id']}: declared size exceeds per-file budget")
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"{entry['id']}: download exceeded per-file budget")
                        stream.write(chunk)
            actual = digest(temporary)
            if expected and actual != expected and not lock:
                raise ValueError(f"{entry['id']}: SHA-256 mismatch; refusing changed upstream data")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    result = {**entry, "sha256": actual, "bytes": target.stat().st_size}
    if entry.get("extract"):
        extract(target, root, limit, verify_only=verify_only)
    print(f"verified {entry['id']}: {result['bytes']:,} bytes", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / ".test-data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/data-sources.json")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-file-mib", type=int, default=128)
    args = parser.parse_args()
    if args.max_file_mib < 1:
        parser.error("--max-file-mib must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.directory.mkdir(parents=True, exist_ok=True)
    entries = manifest["entries"]
    errors = []
    completed = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                fetch,
                entry,
                args.directory,
                lock=args.lock,
                verify_only=args.verify_only,
                limit=args.max_file_mib * 1024 * 1024,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                completed[entry["id"]] = future.result()
            except Exception as exc:
                errors.append(f"{entry['id']}: {exc}")
    if args.lock:
        manifest["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["entries"] = [completed.get(e["id"], e) for e in manifest["entries"]]
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"{len(completed)}/{len(entries)} fixtures verified; {sum(e['bytes'] for e in completed.values()):,} bytes"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

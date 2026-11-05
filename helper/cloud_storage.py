"""Manifest-based S3 storage for reviewed datasets and immutable training runs.

Files are content addressed. Publishing the manifest last makes partial uploads
invisible; conditional writes prevent simultaneous writers losing each other's work.
"""

import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

import yaml
from botocore.exceptions import ClientError

from helper.dataset import prepare_dataset
from helper.snapshot_s3 import file_hash


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Cloud names must use 1-128 letters, digits, underscores, dots or hyphens; start with a letter/digit.")
    return value


def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or relative != "/".join(parts) or any(
        p in {".", ".."} or re.search(r'[<>:"\\|?*\x00-\x1f]', p) or p.endswith((".", " "))
        or p.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        for p in parts
    ):
        raise ValueError(f"Unsafe cloud path: {relative!r}")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Cloud path escapes destination: {relative!r}")
    return path


def read_manifest(client, bucket, prefix):
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")
    except ClientError as error:
        if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return None, None
        raise
    with response["Body"] as body:
        manifest = json.load(body)
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported cloud manifest.")
    normalized = set()
    for relative, entry in manifest["files"].items():
        safe_path(Path.cwd(), relative)
        if relative.casefold() in normalized:
            raise ValueError(f"Cloud paths collide on Windows: {relative}")
        normalized.add(relative.casefold())
        if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) or not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid file fingerprint: {relative}")
    return manifest, response["ETag"]


def publish(client, bucket, prefix, root, paths, metadata, remote=None, etag=None):
    files = dict(remote["files"]) if remote else {}
    pending = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        entry = {"sha256": file_hash(path), "size": path.stat().st_size}
        if relative in files and files[relative] != entry:
            raise ValueError(f"Cloud conflict for {relative}; use a new sample name/dataset or a new run ID.")
        if relative not in files:
            pending.append((path, entry))
        files[relative] = entry
    manifest = {"version": 1, "files": files, **metadata}
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError("Cloud paths collide on Windows; use unique names ignoring case.")
    if manifest.get("kind") == "dataset":
        dataset_manifest(manifest)
    if manifest == remote:
        return manifest
    # Unique hashes cannot change objects used by a previously published manifest.
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "upload"
        for path, entry in pending:
            shutil.copyfile(path, staged)
            if file_hash(staged) != entry["sha256"]:
                raise ValueError(f"File changed during upload: {path}; stop writers and retry.")
            client.upload_file(str(staged), bucket, f"{prefix}/objects/{entry['sha256']}")
    condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
    try:
        client.put_object(Bucket=bucket, Key=f"{prefix}/manifest.json",
                          Body=json.dumps(manifest, sort_keys=True).encode(), ContentType="application/json", **condition)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}:
            raise ValueError("Cloud manifest changed concurrently. Rerun to merge the latest version.") from error
        raise
    return manifest


def restore(client, bucket, prefix, root, manifest, dataset=False):
    # Preflight all conflicts before downloading or modifying any destination files.
    missing = []
    for relative, entry in manifest["files"].items():
        destination = safe_path(root, relative)
        if destination.exists():
            if not destination.is_file() or file_hash(destination) != entry["sha256"]:
                raise ValueError(f"Local conflict for {destination}; move the local copy aside and retry.")
        else:
            missing.append(relative)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        staging = Path(temporary)
        for relative, entry in manifest["files"].items():
            target = safe_path(staging, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in missing:
                client.download_file(bucket, f"{prefix}/objects/{entry['sha256']}", str(target))
            else:
                shutil.copyfile(root / relative, target)
            if target.stat().st_size != entry["size"] or file_hash(target) != entry["sha256"]:
                raise ValueError(f"Cloud checksum mismatch: {relative}")
        if dataset:
            prepare_dataset(staging, **manifest["settings"])
        for relative in missing:
            destination = safe_path(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(destination)


def dataset_manifest(manifest):
    if manifest.get("kind") != "dataset":
        raise ValueError("Expected a dataset manifest.")
    settings = manifest.get("settings", {})
    if set(settings) != {"depth_scale", "max_depth"} or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in settings.values()
    ):
        raise ValueError("Invalid cloud depth settings.")
    files = manifest["files"]
    for relative in files:
        parts = PurePosixPath(relative).parts
        if len(parts) < 3 or parts[0] not in {"images", "depth"} or parts[1] not in {"train", "val"}:
            raise ValueError(f"Unexpected dataset file: {relative}")
        is_rgb = parts[0] == "images"
        if PurePosixPath(relative).suffix != (".jpg" if is_rgb else ".png"):
            raise ValueError(f"Unexpected dataset extension: {relative}")
        partner = str(PurePosixPath("depth" if is_rgb else "images", *parts[1:]).with_suffix(".png" if is_rgb else ".jpg"))
        if partner not in files:
            raise ValueError(f"Incomplete cloud pair: {relative}")


def sync_dataset(client, bucket, root, name="curated", direction="sync", depth_scale=None, max_depth=None):
    if direction not in {"push", "pull", "sync"}:
        raise ValueError(f"Unknown sync direction: {direction}")
    root = Path(root).expanduser().resolve()
    prefix = f"datasets/{identifier(name)}"
    remote, etag = read_manifest(client, bucket, prefix)
    if remote:
        dataset_manifest(remote)
        for key, value in (("depth_scale", depth_scale), ("max_depth", max_depth)):
            if value is not None and value != remote["settings"][key]:
                raise ValueError(f"--{key.replace('_', '-')} differs from cloud settings; use a separate dataset.")
        config = root / "depth_dataset.generated.yaml"
        if config.exists():
            saved = yaml.safe_load(config.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in remote["settings"].items()):
                raise ValueError("Local and cloud depth settings differ; use a separate dataset.")
    if direction in {"pull", "sync"} and remote:
        restore(client, bucket, prefix, root, remote, dataset=True)
    if direction == "pull" and remote is None:
        raise ValueError(f"No curated dataset at s3://{bucket}/{prefix}/")
    config = prepare_dataset(root, **(remote["settings"] if remote else {"depth_scale": depth_scale, "max_depth": max_depth}))
    if direction == "pull":
        return remote
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    paths = sorted(p for folder, suffix in (("images", ".jpg"), ("depth", ".png"))
                   for p in (root / folder).rglob("*") if p.is_file() and p.suffix.lower() == suffix)
    return publish(client, bucket, prefix, root, paths,
                   {"kind": "dataset", "settings": {k: settings[k] for k in ("depth_scale", "max_depth")}}, remote, etag)


def upload_run(client, bucket, root, run_id=None, provenance=None):
    root = Path(root).expanduser().resolve()
    run_id = identifier(run_id or f"{root.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}")
    prefix = f"runs/{run_id}"
    remote, _ = read_manifest(client, bucket, prefix)
    if remote:
        raise ValueError(f"Run {run_id} already exists; choose a new ID to retain both results.")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not paths or not (root / "weights/best.pt").is_file():
        raise ValueError(f"Expected a completed training directory with weights/best.pt: {root}")
    publish(client, bucket, prefix, root, paths, {"kind": "run", "provenance": provenance or {}})
    return run_id


def download_run(client, bucket, root, run_id):
    prefix = f"runs/{identifier(run_id)}"
    manifest, _ = read_manifest(client, bucket, prefix)
    if not manifest or manifest.get("kind") != "run":
        raise ValueError(f"No completed run: {run_id}")
    restore(client, bucket, prefix, Path(root).expanduser().resolve(), manifest)

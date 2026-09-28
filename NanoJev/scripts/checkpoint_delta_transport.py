#!/usr/bin/env python3
"""Lossless checkpoint transport: byte XOR, four byte lanes, gzip level 1.

pack --base BASE --target TARGET --output ARCHIVE
unpack --base BASE --archive ARCHIVE --output TARGET
--self-check

Inputs must have equal lengths. No tensor parsing, quantization, or numerical
conversion occurs. ARCHIVE.json records both input hashes and the archive hash.
Verified temporary files are published with an atomic, no-overwrite hard link.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

FORMAT = "nanojev-checkpoint-xor-lanes4-gzip-v1"
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def sidecar(archive):
    return Path(str(archive) + ".json")


def refuse_existing(path):
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")


def temp_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    return Path(name), os.fdopen(fd, "wb")


def publish(temp, output):
    # Both paths share a directory/filesystem. Unlike replace/rename, link
    # atomically refuses a concurrently created destination as well.
    os.link(temp, output)
    temp.unlink()


def xor_bytes(left, right):
    if len(left) != len(right):
        raise ValueError("XOR blocks must have identical lengths")
    return (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(len(left), "little")


def plane_bytes(block):
    return b"".join(block[lane::4] for lane in range(4))


def unplane_bytes(block):
    result, offset, n = bytearray(len(block)), 0, len(block)
    for lane in range(4):
        count = max(0, (n - lane + 3) // 4)
        result[lane::4] = block[offset:offset + count]
        offset += count
    return bytes(result)


def file_signature(handle):
    stat = os.fstat(handle.fileno())
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def pack(base, target, output, chunk_size=DEFAULT_CHUNK_SIZE):
    base, target, output = map(Path, (base, target, output))
    metadata_path = sidecar(output)
    refuse_existing(output)
    refuse_existing(metadata_path)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    size = target.stat().st_size
    if base.stat().st_size != size:
        raise ValueError("This format requires equal base and target byte lengths")
    temp = metadata_temp = None
    published = False
    try:
        temp, raw = temp_output(output)
        base_hash, target_hash = hashlib.sha256(), hashlib.sha256()
        with base.open("rb") as base_file, target.open("rb") as target_file, raw:
            signatures = (file_signature(base_file), file_signature(target_file))
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
                remaining = size
                while remaining:
                    n = min(chunk_size, remaining)
                    a, b = base_file.read(n), target_file.read(n)
                    if len(a) != n or len(b) != n:
                        raise ValueError("An input was shortened while packing")
                    base_hash.update(a)
                    target_hash.update(b)
                    compressed.write(plane_bytes(xor_bytes(a, b)))
                    remaining -= n
                if base_file.read(1) or target_file.read(1):
                    raise ValueError("An input grew while packing")
            if signatures != (file_signature(base_file), file_signature(target_file)):
                raise ValueError("An input changed while packing")
            raw.flush()
            os.fsync(raw.fileno())
        metadata = {"format": FORMAT, "base_sha256": base_hash.hexdigest(),
            "target_sha256": target_hash.hexdigest(), "base_size": size, "target_size": size,
            "chunk_size": chunk_size, "archive_sha256": sha256_file(temp),
            "archive_size": temp.stat().st_size, "compression": "gzip", "compression_level": 1,
            "transform": "bytewise XOR, then concatenate offsets 0::4, 1::4, 2::4, 3::4 separately in each chunk"}
        metadata_temp, handle = temp_output(metadata_path)
        with handle:
            handle.write((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        publish(temp, output)
        published = True
        publish(metadata_temp, metadata_path)
        return metadata
    except BaseException:
        # A failed pair publication leaves no usable archive without metadata.
        if published:
            output.unlink(missing_ok=True)
        raise
    finally:
        for path in (temp, metadata_temp):
            if path is not None:
                path.unlink(missing_ok=True)


def load_metadata(archive):
    metadata = json.loads(sidecar(archive).read_text())
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint delta format")
    for key in ("base_sha256", "target_sha256", "archive_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {key}")
    for key in ("base_size", "target_size", "archive_size", "chunk_size"):
        value = metadata.get(key)
        if type(value) is not int or value < (1 if key == "chunk_size" else 0):
            raise ValueError(f"Invalid {key}")
    if metadata["base_size"] != metadata["target_size"]:
        raise ValueError("This format requires equal base and target byte lengths")
    if metadata.get("compression") != "gzip" or metadata.get("compression_level") != 1:
        raise ValueError("Unsupported compression contract")
    return metadata


def unpack(base, archive, output):
    base, archive, output = map(Path, (base, archive, output))
    refuse_existing(output)
    metadata = load_metadata(archive)
    if base.stat().st_size != metadata["base_size"]:
        raise ValueError("Base size mismatch")
    if archive.stat().st_size != metadata["archive_size"] or sha256_file(archive) != metadata["archive_sha256"]:
        raise ValueError("Archive SHA256 or size mismatch")
    temp = None
    try:
        temp, handle = temp_output(output)
        base_hash, target_hash = hashlib.sha256(), hashlib.sha256()
        with base.open("rb") as base_file, gzip.open(archive, "rb") as compressed, handle:
            signature = file_signature(base_file)
            remaining = metadata["target_size"]
            while remaining:
                n = min(metadata["chunk_size"], remaining)
                a, planes = base_file.read(n), compressed.read(n)
                if len(a) != n or len(planes) != n:
                    raise ValueError("Short base or decompressed delta")
                target = xor_bytes(a, unplane_bytes(planes))
                base_hash.update(a)
                target_hash.update(target)
                handle.write(target)
                remaining -= n
            # Reading through EOF also checks gzip trailers/CRC and rejects
            # extra decompressed data instead of silently ignoring it.
            if compressed.read(1) or base_file.read(1):
                raise ValueError("Unexpected bytes after the declared target size")
            if signature != file_signature(base_file):
                raise ValueError("Base changed while unpacking")
            if base_hash.hexdigest() != metadata["base_sha256"]:
                raise ValueError("Base SHA256 mismatch")
            if target_hash.hexdigest() != metadata["target_sha256"]:
                raise ValueError("Recovered target SHA256 mismatch")
            handle.flush()
            os.fsync(handle.fileno())
        # Re-read the temporary output before publication to verify the actual
        # bytes on disk, in addition to the streamed reconstruction digest.
        if temp.stat().st_size != metadata["target_size"] or sha256_file(temp) != metadata["target_sha256"]:
            raise ValueError("Written target SHA256 or size mismatch")
        publish(temp, output)
        return {"format": FORMAT, "verified": True, "base_sha256": metadata["base_sha256"],
                "target_sha256": metadata["target_sha256"], "target_size": metadata["target_size"],
                "archive_sha256": metadata["archive_sha256"]}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def self_check():
    rng = random.Random(20260919)
    cases = [(b"", b"", 4), (bytes(4097), bytes(4097), 1024),
             (rng.randbytes(65543), rng.randbytes(65543), 4096),
             (rng.randbytes(31), rng.randbytes(31), 7)]
    cases += [(rng.randbytes(n), rng.randbytes(n), 4) for n in range(1, 6)]
    passed = []
    with tempfile.TemporaryDirectory(prefix="nanojev_delta_check_") as folder:
        root = Path(folder)
        for i, (a, b, chunk) in enumerate(cases):
            base, target, archive, recovered = [root / f"{i}.{name}" for name in ("base", "target", "gz", "recovered")]
            base.write_bytes(a)
            target.write_bytes(b)
            metadata = pack(base, target, archive, chunk)
            unpack(base, archive, recovered)
            assert recovered.read_bytes() == b and metadata["target_sha256"] == hashlib.sha256(b).hexdigest()
            assert unplane_bytes(plane_bytes(b)) == b
            passed.append(f"roundtrip_size_{len(b)}_chunk_{chunk}")
        base, target, archive = root / "base", root / "target", root / "archive.gz"
        base.write_bytes(rng.randbytes(1025))
        target.write_bytes(rng.randbytes(1025))
        pack(base, target, archive, 128)

        def rejects(name, fn, output):
            try:
                fn()
            except (ValueError, OSError, EOFError):
                assert not output.exists()
                passed.append(name)
            else:
                raise AssertionError(f"Expected rejection: {name}")

        bad_base = root / "bad.base"
        bad_base.write_bytes(bytes(1025))
        output = root / "bad.base.output"
        rejects("wrong_base_hash", lambda: unpack(bad_base, archive, output), output)
        damaged = root / "damaged.gz"
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 1
        damaged.write_bytes(data)
        sidecar(damaged).write_bytes(sidecar(archive).read_bytes())
        output = root / "damaged.output"
        rejects("damaged_archive_hash", lambda: unpack(base, damaged, output), output)
        metadata = json.loads(sidecar(damaged).read_text())
        metadata["archive_sha256"] = sha256_file(damaged)
        sidecar(damaged).write_text(json.dumps(metadata))
        rejects("damaged_gzip_even_with_rehashed_archive", lambda: unpack(base, damaged, output), output)
        metadata = json.loads(sidecar(archive).read_text())
        metadata["target_sha256"] = "0" * 64
        sidecar(archive).write_text(json.dumps(metadata))
        output = root / "wrong.target.output"
        rejects("wrong_target_hash", lambda: unpack(base, archive, output), output)
        output = root / "unequal.gz"
        rejects("unequal_size", lambda: pack(base, root / "0.target", output), output)
        original = target.read_bytes()
        try:
            unpack(base, archive, target)
        except FileExistsError:
            assert target.read_bytes() == original
            passed.append("existing_target_preserved")
        else:
            raise AssertionError("Existing output was overwritten")
        assert not list(root.glob(".*.tmp")), "Temporary files were not cleaned up"
    return {"self_check_passed": True, "checks": passed, "real_checkpoint_processing": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    commands = parser.add_subparsers(dest="command")
    packing = commands.add_parser("pack")
    packing.add_argument("--base", type=Path, required=True)
    packing.add_argument("--target", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    unpacking = commands.add_parser("unpack")
    unpacking.add_argument("--base", type=Path, required=True)
    unpacking.add_argument("--archive", type=Path, required=True)
    unpacking.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.self_check:
        if args.command:
            parser.error("--self-check is a separate mode")
        result = self_check()
    elif args.command == "pack":
        result = pack(args.base, args.target, args.output, args.chunk_size)
    elif args.command == "unpack":
        result = unpack(args.base, args.archive, args.output)
    else:
        parser.error("Choose pack, unpack or --self-check")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

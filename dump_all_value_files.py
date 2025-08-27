#!/usr/bin/env python3
"""
Full Value File Dump Utility

Generates core_logic_dump_full.txt: aggregated text of all value-creating source/config/doc/test files.
Excludes: .venv, logs, data payloads, caches, large/binary files.

Env knobs:
  DUMP_MAX_LINES_PER_FILE (default 4000)
  DUMP_MAX_BYTES_PER_FILE (default 1000000)
"""

from __future__ import annotations
import os, sys, datetime, hashlib

INCLUDE_EXT = {".py", ".md", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".txt"}
INCLUDE_DIR_HINTS = {"v26meme", "configs", "dashboard", "scripts", "tests", ".github"}
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "logs", "data", ".idea",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build"}
EXCLUDE_SUFFIXES = {".parquet", ".arrow", ".feather", ".pkl", ".bin", ".so", ".dll"}

OUTPUT = "core_logic_dump_full.txt"
MAX_LINES = int(os.environ.get("DUMP_MAX_LINES_PER_FILE", "4000"))
MAX_BYTES = int(os.environ.get("DUMP_MAX_BYTES_PER_FILE", "1000000"))

def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

def is_text_candidate(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in INCLUDE_EXT

def iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if rel_dir.startswith(".git"):
            continue
        for fn in filenames:
            rel = os.path.join(rel_dir, fn) if rel_dir != "." else fn
            if any(rel.endswith(suf) for suf in EXCLUDE_SUFFIXES):
                continue
            top = rel.split(os.sep)[0]
            if top not in INCLUDE_DIR_HINTS and not is_text_candidate(rel):
                continue
            yield rel

def dump_one(abs_path: str):
    try:
        size = os.path.getsize(abs_path)
        if size > MAX_BYTES:
            with open(abs_path, "rb") as f:
                head = f.read(2048)
            return [f"[SKIPPED large size={size} sha12={sha12(head)}]\n"]
        with open(abs_path, "rb") as f:
            raw = f.read()
        if b"\x00" in raw[:2048]:
            return [f"[SKIPPED binary-like sha12={sha12(raw[:2048])} size={len(raw)}]\n"]
        text = raw.decode("utf-8", "replace").splitlines(keepends=True)
        truncated = False
        if len(text) > MAX_LINES:
            text = text[:MAX_LINES]
            truncated = True
        lines = [f"[lines={len(text)} truncated={truncated} size={len(raw)} sha12={sha12(raw)}]\n"]
        for i, line in enumerate(text, 1):
            lines.append(f"{i:04d}: {line}")
        if truncated:
            lines.append(f"\n[TRUNCATED at {MAX_LINES} lines]\n")
        return lines
    except Exception as e:
        return [f"[ERROR reading file: {e}]\n"]

def main() -> int:
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    files = sorted(iter_files(repo_root))
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out_path = os.path.join(repo_root, OUTPUT)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=== FULL CORE VALUE LOGIC DUMP ===\n")
        out.write(f"GeneratedUTC: {ts}\n")
        out.write(f"MAX_LINES_PER_FILE={MAX_LINES} MAX_BYTES_PER_FILE={MAX_BYTES}\n")
        out.write(f"FileCount={len(files)}\n\n")
        for rel in files:
            abs_path = os.path.join(repo_root, rel)
            out.write(f"\n--- FILE: {rel} ---\n")
            for line in dump_one(abs_path):
                out.write(line)
    print(f"Wrote {OUTPUT} ({len(files)} files)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
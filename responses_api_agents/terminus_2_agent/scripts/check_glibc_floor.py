# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail when the in-sandbox agent import loads a native library that needs a newer glibc than task images have.

The agent runs inside the task image against that image's libc, so every shared object it maps from the agent
deps prefix must load on the oldest supported task image (glibc 2.17, the manylinux2014 floor).

usage: python check_glibc_floor.py MODULE [--floor 2.17] [--optional DIST ...]
"""

import argparse
import importlib
import os
import struct
import sys
from pathlib import Path


SHT_GNU_VERNEED = 0x6FFFFFFE


def required_glibc(path: str) -> list[tuple[int, ...]]:
    """GLIBC_x.y symbol versions an ELF shared object requires (its .gnu.version_r entries)."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"{path} is not an ELF file")
    wide = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if wide:
        shoff, shentsize, shnum = (
            struct.unpack_from(end + "Q", data, 0x28)[0],
            *struct.unpack_from(end + "HH", data, 0x3A),
        )
        section = struct.Struct(end + "IIQQQQIIQQ")
    else:
        shoff, shentsize, shnum = (
            struct.unpack_from(end + "I", data, 0x20)[0],
            *struct.unpack_from(end + "HH", data, 0x2E),
        )
        section = struct.Struct(end + "IIIIIIIIII")
    sections = [section.unpack_from(data, shoff + i * shentsize) for i in range(shnum)]
    versions = set()
    for _name, kind, _flags, _addr, offset, _size, link, count, *_ in sections:
        if kind != SHT_GNU_VERNEED:
            continue
        strtab = sections[link][4]
        need = offset
        for _ in range(count):
            _version, aux_count, _file, aux, nxt = struct.unpack_from(end + "HHIII", data, need)
            entry = need + aux
            for _ in range(aux_count):
                _hash, _aux_flags, _other, name, aux_next = struct.unpack_from(end + "IHHII", data, entry)
                label = data[strtab + name : data.index(b"\0", strtab + name)].decode()
                if label.startswith("GLIBC_") and label[6:7].isdigit():
                    versions.add(tuple(int(part) for part in label[6:].split(".")))
                entry += aux_next
            need += nxt
    return sorted(versions)


def mapped_libraries(prefix: str) -> list[str]:
    """Shared objects under prefix that this process has mapped."""
    root = os.path.realpath(prefix) + os.sep
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and ".so" in fields[5] and os.path.realpath(fields[5].strip()).startswith(root):
                paths.add(fields[5].strip())
    return sorted(paths)


def offenders(prefix: str, floor: tuple[int, ...], optional: tuple[str, ...] = ()) -> list[str]:
    """Mapped libraries needing a glibc above floor, except those inside an optional package directory."""
    root = Path(os.path.realpath(prefix))
    found = []
    for path in mapped_libraries(prefix):
        needed = required_glibc(path)
        if not needed or needed[-1] <= floor:
            continue
        if set(Path(os.path.realpath(path)).relative_to(root).parts[:-1]) & set(optional):
            continue
        found.append(f"{path} requires GLIBC_{'.'.join(map(str, needed[-1]))}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--floor", default="2.17")
    parser.add_argument("--optional", nargs="*", default=(), help="top-level packages whose import failure is handled")
    args = parser.parse_args()
    importlib.import_module(args.module)
    floor = tuple(int(part) for part in args.floor.split("."))
    bad = offenders(sys.prefix, floor, tuple(args.optional))
    if bad:
        sys.exit(f"agent deps need a glibc newer than {args.floor}:\n" + "\n".join(bad))
    print(f"agent deps load on glibc {args.floor}: {args.module}")


if __name__ == "__main__":
    main()

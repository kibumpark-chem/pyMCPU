#!/usr/bin/env python3
"""Check that the pair-search layer is fully inlined into the hot functions.

The walk drivers in ``include/pymcpu/neighbor/`` (``PairSearch.h``,
``SpanMask.h``, ``MovedCells.h``) are always-inlined templates that take
their callbacks as forwarding references. The term's hot function is meant
to be the only frame: a call per probe, per cell or per pair to a driver or
to a lambda's ``operator()`` costs a measurable share of the step, and
unrelated edits have moved the LTO inliner's decisions before (scan speed
swung by up to 1.7x). This script reads the built extension and fails if
any hot function calls such a symbol::

    python scripts/check_inlining.py [--so path/to/mcpu_core*.so] [--import-root DIR]

How: ``nm -C -S`` gives each hot function's address and size (every clone
counts too, e.g. ``[clone .cold]``), ``objdump -d -C`` disassembles exactly
those ranges, and every ``call`` or ``jmp`` to a named symbol is checked.
A target is a violation when the called function itself (its name with
template arguments removed) is in ``mcpu::neighbor::`` or is a lambda's
``operator()``. An out-of-line template whose arguments merely mention a
layer callback (an ``OpenCellGrid`` walk instantiated with one, say) is a
grid frame, not a layer frame: it is reported as a note, not a failure.
Calls to anything else (``std::vector`` growth, libm) are listed with
``-v`` but allowed.

Exit status: 0 when every hot function was found and is clean, 1 on a
violation, 2 when the library or a hot function cannot be found. Needs
binutils (``nm``, ``objdump``) and a build that keeps its symbol table.
pybind11 strips a Release extension when it links it, so configure the
Release build with ``-DCMAKE_STRIP=/bin/true``. RelWithDebInfo keeps the
symbols but builds without LTO, so it is not the binary that ships.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

# Demangled-name substrings of the hot functions to inspect. Each must match
# at least one symbol, or the check cannot vouch for it (exit 2).
HOT_FUNCTIONS = (
    "MuPotential::calculateEnergyChange_clist(",
)

# A called function is a frame the layer should not add when its own name
# is in the layer's namespace or it is a lambda's call operator. MENTIONS
# flags calls whose template or parameter types merely name one.
LAYER_NS = "mcpu::neighbor::"
MENTIONS = ("mcpu::neighbor::", "{lambda(")

_OPERATORS = ("operator<<=", "operator>>=", "operator<<", "operator>>",
              "operator<=>", "operator<=", "operator>=", "operator->",
              "operator<", "operator>")


def own_name(demangled: str) -> str:
    """The demangled name with every template argument list removed."""
    s = demangled
    for k, op in enumerate(_OPERATORS):
        s = s.replace(op, f"operator@{k}@")
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def forbidden(demangled: str) -> bool:
    """True for a call into mcpu::neighbor:: or to a lambda's operator()."""
    s = own_name(demangled)
    if "}::operator()(" in s:  # X::{lambda(...)#1}::operator()(...)
        return True
    head, depth = [], 0  # the qualified name, up to its parameter list
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "(" and depth == 0:
            break
        head.append(ch)
    # The callee is the last token; anything before it is a return type
    # (std::vector<neighbor::X>::emplace_back returns a neighbor::X&).
    parts = "".join(head).split()
    return bool(parts) and parts[-1].startswith(LAYER_NS)

CALL_RE = re.compile(r"\b(call|callq|jmp|jmpq)\s+[0-9a-f]+\s+<(.+)>\s*$")


def find_so(import_root: str | None) -> str:
    roots = [import_root] if import_root else []
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    for root in roots:
        hits = sorted(glob.glob(os.path.join(root, "pymcpu", "mcpu_core*.so")))
        if hits:
            return hits[0]
    sys.exit("check_inlining: no pymcpu/mcpu_core*.so found; pass --so")


def hot_symbols(so: str, patterns: tuple[str, ...]):
    out = subprocess.run(["nm", "-C", "-S", "--defined-only", so],
                         check=True, capture_output=True, text=True).stdout
    found = {p: [] for p in patterns}
    for line in out.splitlines():
        parts = line.split(" ", 3)
        if len(parts) != 4 or parts[2] not in "tTwW":
            continue
        addr, size, _, name = parts
        for p in patterns:
            if p in name:
                found[p].append((int(addr, 16), int(size, 16), name))
    return found


def calls_in(so: str, start: int, size: int):
    out = subprocess.run(
        ["objdump", "-d", "-C", "--no-show-raw-insn",
         f"--start-address={start:#x}", f"--stop-address={start + size:#x}", so],
        check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = CALL_RE.search(line)
        if m:
            yield m.group(1), m.group(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--so", help="the built mcpu_core extension")
    ap.add_argument("--import-root", help="checkout whose pymcpu/ holds the .so")
    ap.add_argument("--function", action="append", default=None,
                    help="demangled-name substring of a hot function "
                         "(repeatable; replaces the default list)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also list the allowed calls")
    args = ap.parse_args()
    so = args.so or find_so(args.import_root)
    patterns = tuple(args.function) if args.function else HOT_FUNCTIONS
    found = hot_symbols(so, patterns)

    status = 0
    for p in patterns:
        syms = found[p]
        if not syms:
            print(f"MISSING  no symbol matches {p!r} (inlined into a caller "
                  f"or stripped?); pass --function for the caller")
            status = max(status, 2)
            continue
        for addr, size, name in syms:
            bad, notes, ok = [], [], []
            for insn, target in calls_in(so, addr, size):
                base = target.split("+0x")[0]
                if base == name:
                    continue  # a branch within this function
                if forbidden(base):
                    bad.append(f"{insn} {target}")
                elif any(f in base for f in MENTIONS):
                    notes.append(f"{insn} {own_name(base)}")
                else:
                    ok.append(f"{insn} {target}")
            short = name if len(name) < 110 else name[:107] + "..."
            print(f"{'FAIL' if bad else 'OK  '}  {short}  "
                  f"({size} bytes, {len(bad) + len(notes) + len(ok)} calls, "
                  f"{len(bad)} into the layer or a lambda)")
            for c in bad:
                print(f"    forbidden: {c[:200]}")
            for c in notes:
                print(f"    note: out-of-line template with a callback: "
                      f"{c[:160]}")
            if args.verbose:
                for c in sorted(set(ok)):
                    print(f"    allowed:   {c[:200]}")
            if bad:
                status = max(status, 1)
    print(f"check_inlining: {'PASS' if status == 0 else 'FAIL'} ({so})")
    return status


if __name__ == "__main__":
    sys.exit(main())

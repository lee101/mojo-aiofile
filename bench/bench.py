"""Correctness-gated benchmark for mojo-aiofile.

Every case checks agreement with the real aiofile function before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_aiofile as maf  # noqa: E402
from aiofile.aio import parse_mode  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_parse_mode(n: int = 1 << 16):
    """Mode parsing, one call each, against `aiofile.aio.parse_mode`.

    A single mode string is a handful of OR-ed flags, so the honest comparison
    is one string per call in both.
    """
    modes = ["r", "rb", "w+", "ab+", "xb", "r+b", "br", "rbt"] * (n // 8)

    def run_python():
        acc = 0
        for m in modes:
            acc += parse_mode(m).flags
        return acc

    def run_mojo():
        acc = 0
        for m in modes:
            acc += maf.parse_mode(m).flags
        return acc

    assert run_python() == run_mojo()
    return f"parse_mode n={n}", _time(run_python, 3), _time(run_mojo, 3)


def bench_parse_mode_one_long_mode(n: int = 1 << 16):
    """The same kernel over one long mode string.

    `parse_mode` is O(len(mode)) and a real mode string is one to four
    characters, so this case is a lower bound on the per-byte cost rather than
    a realistic workload. It is here to show the loop itself, isolated from the
    transition.
    """
    from mojo_aiofile._lib import lib, OS_FLAGS

    # a long run of 'b' is a valid mode (binary set repeatedly), so the kernel
    # walks every byte; no real mode string is this long
    mode = b"b" * n
    buf = np.frombuffer(mode, dtype=np.uint8)
    out = np.zeros(8, dtype=np.int64)
    assert maf.parse_mode("b" * 8).flags == parse_mode("b" * 8).flags
    addr, oaddr = buf.ctypes.data, out.ctypes.data

    def one():
        lib.af_parse_mode(addr, buf.size, *OS_FLAGS, oaddr)

    def run_python():
        return parse_mode(mode.decode()).flags

    return f"parse_mode one long mode n={n}", _time(run_python, 3), _time(one, 3)


def bench_running_offsets(n: int = 1 << 18):
    """Running read offsets, against the same accumulation in Python."""
    rng = np.random.default_rng(0)
    sizes = rng.integers(0, 65536, n, dtype=np.int64)

    def run_python():
        acc = 0
        for s in sizes:
            acc += int(s)
        return acc

    def run_mojo():
        _offsets, final = maf.running_offsets(sizes, 0)
        return final

    assert run_python() == run_mojo()
    return f"running_offsets n={n}", _time(run_python, 3), _time(run_mojo, 3)


def bench_find_sep(n: int = 1 << 22):
    """Separator scan, against `bytes.find` -- the fastest fair equivalent."""
    buf = np.zeros(n, dtype=np.uint8)
    raw = buf.tobytes()
    raw = raw[:-1] + b"\n"

    assert maf.find_sep(raw, b"\n") == raw.find(b"\n")
    assert maf.find_sep(raw[:-1], b"\n") == -1

    def run_python():
        return raw.find(b"\n")

    def run_mojo():
        return maf.find_sep(raw, b"\n")

    return f"find_sep n={n}", _time(run_python), _time(run_mojo)


def bench_find_sep_lines(n: int = 1 << 22, line_len: int = 64):
    """One scan per line, which is what `LineReader` actually does."""
    raw = (b"x" * (line_len - 1) + b"\n") * (n // line_len)
    assert maf.find_sep(raw, b"\n") == raw.find(b"\n")

    def run_python():
        pos = 0
        acc = 0
        while True:
            pos = raw.find(b"\n", pos)
            if pos < 0:
                break
            acc += pos
            pos += 1
        return acc

    def run_mojo():
        pos = 0
        acc = 0
        while True:
            pos = maf.find_sep(raw, b"\n", pos)
            if pos < 0:
                break
            acc += pos
            pos += 1
        return acc

    assert run_python() == run_mojo()
    return f"find_sep per line {n}B", _time(run_python), _time(run_mojo)


def main():
    print(f"{'case':<34}{'reference':>13}{'mojo-aiofile':>16}{'ratio':>10}")
    print("-" * 73)
    for fn in (
        bench_parse_mode,
        bench_parse_mode_one_long_mode,
        bench_running_offsets,
        bench_find_sep,
        bench_find_sep_lines,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref*1e3:>11.2f}ms{got*1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()

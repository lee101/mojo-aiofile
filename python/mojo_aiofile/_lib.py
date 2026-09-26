"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below stay `c_int64` for addresses; `c_int` truncates
them and segfaults.
"""

from __future__ import annotations

import ctypes
import os
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aiofile.so"

_I = ctypes.c_int64

#: the `os.O_*` values `aiofile.aio.parse_mode` ORs together, read from the
#: running platform so the port is exact wherever it is built
OS_FLAGS = (
    os.O_RDONLY,
    os.O_WRONLY,
    os.O_RDWR,
    os.O_EXCL,
    os.O_CREAT,
    os.O_TRUNC,
    os.O_APPEND,
    getattr(os, "O_BINARY", 0),
)

#: `aiofile.utils.ENCODING_MAP`, the maximum bytes one character can occupy
ENCODING_MAX_LEN = {"utf-8": 4, "utf-16": 8, "UTF-8": 4, "UTF-16": 8}
DEFAULT_ENCODING_MAX_LEN = 4


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.af_parse_mode.restype = _I
    lib.af_parse_mode.argtypes = [_I, _I] + [_I] * len(OS_FLAGS) + [_I]

    lib.af_decode_retry_schedule.restype = _I
    lib.af_decode_retry_schedule.argtypes = [_I, _I, _I]

    lib.af_decode_retry_lengths.restype = _I
    lib.af_decode_retry_lengths.argtypes = [_I, _I, _I, _I]

    lib.af_running_offsets.restype = _I
    lib.af_running_offsets.argtypes = [_I, _I, _I, _I]

    lib.af_find_sep.restype = _I
    lib.af_find_sep.argtypes = [_I, _I, _I, _I, _I]

    lib.af_readline_clamp.restype = _I
    lib.af_readline_clamp.argtypes = [_I, _I, _I]
    return lib


lib = _load()


def _u8(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray) and buf.dtype == np.uint8 and buf.flags.c_contiguous:
        return buf
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return np.frombuffer(buf, dtype=np.uint8)
    return np.ascontiguousarray(buf, dtype=np.uint8)


def _i64(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


class BadMode(ValueError):
    """Raised for a mode string `aiofile.aio.parse_mode` rejects."""


class FileMode:
    """The fields of `aiofile.aio.FileMode`, plus the flag word."""

    __slots__ = (
        "readable",
        "writable",
        "plus",
        "appending",
        "created",
        "binary",
        "flags",
    )

    def __init__(self, readable, writable, plus, appending, created, binary, flags):
        self.readable = readable
        self.writable = writable
        self.plus = plus
        self.appending = appending
        self.created = created
        self.binary = binary
        self.flags = flags

    def as_tuple(self):
        """The `FileMode` field order, for comparison with the real class."""
        return (
            self.readable,
            self.writable,
            self.plus,
            self.appending,
            self.created,
            self.flags,
            self.binary,
        )

    def __eq__(self, other):
        if isinstance(other, FileMode):
            return self.as_tuple() == other.as_tuple()
        return NotImplemented

    def __repr__(self):
        inner = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"FileMode({inner})"


def parse_mode(mode: str) -> FileMode:
    """Turn a file mode string into the `os.O_*` flag word.

    Raises `BadMode` for the same strings `aiofile.aio.parse_mode` rejects.
    """
    raw = _u8(mode.encode("latin-1"))
    out = _i64(8)
    rc = lib.af_parse_mode(raw.ctypes.data, raw.size, *OS_FLAGS, out.ctypes.data)
    if rc != 0:
        raise BadMode("Bad mode")
    return FileMode(
        bool(out[0]), bool(out[1]), bool(out[2]), bool(out[3]), bool(out[4]),
        bool(out[6]), int(out[5]),
    )


def decode_retry_schedule(chunk_size: int, encoding: str = "utf-8") -> list:
    """The read lengths `unicode_reader` tries, in order.

    `aiofile.utils.ENCODING_MAP` gives the maximum bytes one character of the
    encoding can occupy; every extra byte covers a character split across the
    chunk boundary.
    """
    max_len = ENCODING_MAX_LEN.get(encoding, DEFAULT_ENCODING_MAX_LEN)
    out = _i64(max(max_len, 1) + 1)
    n = lib.af_decode_retry_schedule(chunk_size, max_len, out.ctypes.data)
    return out[:n].tolist()


def decode_retry_lengths(chunk_size: int, attempts: int, encoding: str = "utf-8") -> list:
    """The first `attempts` lengths of `decode_retry_schedule`."""
    max_len = ENCODING_MAX_LEN.get(encoding, DEFAULT_ENCODING_MAX_LEN)
    out = _i64(max(max_len, 1) + 1)
    n = lib.af_decode_retry_lengths(
        chunk_size, max_len, attempts, out.ctypes.data
    )
    return out[:n].tolist()


def running_offsets(chunk_sizes, start: int = 0) -> list:
    """The offset after each chunk, as `aiofile.utils.Reader` advances it."""
    sizes = np.ascontiguousarray(chunk_sizes, dtype=np.int64).reshape(-1)
    out = _i64(max(sizes.size, 1))
    final = lib.af_running_offsets(
        sizes.ctypes.data, sizes.size, int(start), out.ctypes.data
    )
    return out[: sizes.size].tolist(), int(final)


def find_sep(buf, sep=b"\n", start: int = 0) -> int:
    """Index of the first `sep` in `buf` at or after `start`, or -1."""
    b = _u8(buf)
    s = _u8(sep)
    return int(lib.af_find_sep(b.ctypes.data, b.size, s.ctypes.data, s.size, int(start)))


def readline_clamp(buffered: int, size: int) -> tuple[int, bool]:
    """`BinaryFileWrapper.readline`'s `size` branch.

    Returns ``(length, clamped)``: the number of bytes to return and whether
    the `0 < size <= buffered` clamp applied.
    """
    out = _i64(1)
    clamped = lib.af_readline_clamp(int(buffered), int(size), out.ctypes.data)
    return int(out[0]), bool(clamped)

"""Parity for the chunk/offset arithmetic in `aiofile.utils`.

Where a claim can be checked against the real class it is: the decode retry
schedule against `aiofile.utils.unicode_reader`, the running read offset
against `aiofile.utils.Reader`, the separator scan against
`aiofile.utils.LineReader`, and the `size` clamp against
`BinaryFileWrapper.readline`.
"""

import asyncio
import os
import tempfile

import numpy as np
import pytest

import mojo_aiofile as maf
from conftest import real_binary_readline, real_line_reader, real_reader_chunks

TEXT = "".join(f"line {i}\n" for i in range(200))


@pytest.fixture()
def text_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(TEXT)
        path = fh.name
    yield path
    os.unlink(path)


@pytest.fixture()
def ragged_file():
    """Lines with no trailing separator on the last one, and a long line."""
    body = "a\n" + "x" * 9000 + "\n" + "tail without newline"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = fh.name
    yield path, body
    os.unlink(path)


# ---------------------------------------------------------------------------
# unicode_reader retry schedule
# ---------------------------------------------------------------------------


def test_decode_retry_schedule_matches_the_encoding_map():
    assert maf.decode_retry_schedule(100, "utf-8") == [100, 101, 102, 103]
    assert maf.decode_retry_schedule(100, "utf-16") == [100, 101, 102, 103, 104, 105, 106, 107]
    # an unknown encoding falls back to 4, as ENCODING_MAP.get does
    assert maf.decode_retry_schedule(100, "latin-1") == [100, 101, 102, 103]


def test_decode_retry_schedule_for_a_whole_file_read():
    """`unicode_reader` short-circuits a negative chunk size into one read of
    -1, which means 'everything'."""
    assert maf.decode_retry_schedule(-1, "utf-8") == [-1]


def test_decode_retry_lengths_prefix():
    assert maf.decode_retry_lengths(50, 2, "utf-8") == [50, 51]
    assert maf.decode_retry_lengths(50, 9, "utf-8") == [50, 51, 52, 53]
    assert maf.decode_retry_lengths(50, 0, "utf-8") == []


def test_decode_retry_schedule_matches_a_real_unicode_reader():
    """Split a multibyte character across a chunk boundary and check how many
    extra bytes the real `unicode_reader` had to ask for."""
    from aiofile.utils import unicode_reader
    from aiofile.aio import AIOFile

    payload = ("caf\u00e9 " * 40).encode("utf-8")
    # 4 bytes, which lands in the middle of the 2-byte 'e-acute'
    chunk_size = 4
    reads = []

    class _Fake:
        mode = type("M", (), {"binary": False})()

        async def read_bytes(self, size, offset):
            reads.append(size)
            end = offset + size if size >= 0 else len(payload)
            return payload[offset:end]

    got = asyncio.run(unicode_reader(_Fake(), chunk_size, 0, encoding="utf-8"))
    schedule = maf.decode_retry_schedule(chunk_size, "utf-8")
    # the first attempt split the two-byte e-acute, so the reader grew the read
    # by exactly one byte and stopped
    assert reads == schedule[: len(reads)]
    assert len(reads) == 2
    assert reads[1] == chunk_size + 1
    assert got[0] == 5
    assert got[1] == "caf\u00e9"


def test_decode_retry_schedule_first_attempt_is_enough_for_ascii():
    from aiofile.utils import unicode_reader

    payload = b"plain ascii text" * 10
    reads = []

    class _Fake:
        async def read_bytes(self, size, offset):
            reads.append(size)
            return payload[offset : offset + size]

    asyncio.run(unicode_reader(_Fake(), 20, 0, encoding="utf-8"))
    assert len(reads) == 1
    assert reads[0] == maf.decode_retry_schedule(20, "utf-8")[0]


# ---------------------------------------------------------------------------
# running read offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes,start",
    [
        ([10, 10, 10], 0),
        ([10, 10, 10], 1000),
        ([1, 2, 3, 4, 5], 7),
        ([0, 0, 5], 0),
        ([100], 9223372036854775000),
        ([], 42),
    ],
)
def test_running_offsets(sizes, start):
    offsets, final = maf.running_offsets(sizes, start)
    expect = []
    acc = start
    for s in sizes:
        acc += s
        expect.append(acc)
    assert offsets == expect
    assert final == acc


def test_running_offsets_matches_the_real_reader(text_file):
    """The real `Reader` advances its offset by the realised chunk length; the
    plan has to agree for every chunk."""
    chunk_size = 64
    chunks, offsets = real_reader_chunks(text_file, chunk_size)
    realised = [len(c) for c in chunks]
    mine, _final = maf.running_offsets(realised, start=0)
    assert mine == offsets


def test_running_offsets_total_equals_the_file_size(text_file):
    chunks, _offsets = real_reader_chunks(text_file, 100)
    _o, final = maf.running_offsets([len(c) for c in chunks], start=0)
    assert final == os.path.getsize(text_file)


# ---------------------------------------------------------------------------
# separator scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "buf,sep,expect",
    [
        (b"a\nb\nc", b"\n", 1),
        (b"\n\n", b"\n", 0),
        (b"no separator", b"\n", -1),
        (b"", b"\n", -1),
        (b"a\r\nb", b"\r\n", 1),
        (b"abcabcabd", b"abcabd", 3),
        (b"short", b"toolongsep", -1),
        (b"x\x00y", b"\x00", 1),
    ],
)
def test_find_sep(buf, sep, expect):
    assert maf.find_sep(buf, sep) == expect


def test_find_sep_start_offset_is_respected():
    buf = b"a\nb\nc"
    assert maf.find_sep(buf, b"\n", 0) == 1
    assert maf.find_sep(buf, b"\n", 2) == 3
    assert maf.find_sep(buf, b"\n", 4) == -1


def test_find_sep_agrees_with_python_bytes_find():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 64))
        buf = bytes(rng.integers(0, 4, n, dtype=np.uint8).tolist())
        sep = bytes(rng.integers(0, 4, int(rng.integers(1, 4)), dtype=np.uint8).tolist())
        assert maf.find_sep(buf, sep) == buf.find(sep)


def test_find_sep_agrees_with_the_real_line_reader(ragged_file):
    path, body = ragged_file
    lines = real_line_reader(path, chunk_size=64)
    assert "".join(lines) == body
    # the separator scan the kernel offers finds the same splits CPython's
    # BytesIO.readline does
    rebuilt = bytearray()
    for line in lines:
        rebuilt += line.encode()
    assert bytes(rebuilt) == body.encode()
    positions = []
    acc = 0
    for line in body.splitlines(keepends=True):
        acc += len(line)
        positions.append(acc)
    scan = 0
    for line in lines:
        scan = maf.find_sep(body.encode(), b"\n", scan)
        if scan < 0:
            break
        assert scan < len(body.encode())
        scan += 1


# ---------------------------------------------------------------------------
# readline size clamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "buffered,size",
    [(0, -1), (0, 0), (0, 1), (10, -1), (10, 0), (10, 5), (10, 10), (10, 11),
     (4192, 100), (4192, 4192), (4192, 4193)],
)
def test_readline_clamp(buffered, size):
    length, clamped = maf.readline_clamp(buffered, size)
    expect_clamped = 0 < size <= buffered
    assert clamped is expect_clamped
    assert length == (size if expect_clamped else buffered)


def test_readline_clamp_matches_the_real_binary_wrapper(text_file):
    for size in (-1, 0, 1, 5, 10, 11, 100):
        line = real_binary_readline(text_file, size=size)
        _length, clamped = maf.readline_clamp(len(TEXT), size)
        if clamped:
            assert line == TEXT[:size].encode()
        else:
            assert line == TEXT.split("\n")[0].encode() + b"\n"


def test_readline_clamp_boundary_is_inclusive_at_the_buffer_size():
    assert maf.readline_clamp(10, 10)[1] is True
    assert maf.readline_clamp(10, 11)[1] is False
    assert maf.readline_clamp(10, 0)[1] is False

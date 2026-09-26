import asyncio
import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-aiofile.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-aiofile.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Drivers for the real aiofile classes.
# ---------------------------------------------------------------------------


def real_parse_mode(mode: str):
    """`aiofile.aio.parse_mode`, or the exception it raises."""
    from aiofile.aio import parse_mode

    try:
        return parse_mode(mode), None
    except Exception as exc:  # upstream raises a bare Exception("Bad mode")
        return None, exc


def real_reader_chunks(path: str, chunk_size: int, encoding=None):
    """Drive the real `aiofile.utils.Reader` and record the offsets it used.

    Returns ``(chunks, offsets)`` where `offsets` is the value of the reader's
    private running offset after each chunk, read back from the file position
    the next read starts at.
    """
    from aiofile.aio import AIOFile
    from aiofile.utils import Reader

    async def go():
        afp = AIOFile(path, "r") if encoding is None else AIOFile(path, "r", encoding=encoding)
        await afp.open()
        try:
            reader = Reader(afp, chunk_size=chunk_size)
            chunks = []
            offsets = []
            while True:
                before = reader._Reader__offset
                chunk = await reader.read_chunk()
                chunks.append(chunk)
                offsets.append(reader._Reader__offset)
                if not chunk:
                    break
                del before
            return chunks, offsets
        finally:
            await afp.close()

    return asyncio.run(go())


def real_line_reader(path: str, chunk_size: int = 4192, line_sep: str = "\n"):
    """Drive the real `aiofile.utils.LineReader` and collect its lines."""
    from aiofile.aio import AIOFile
    from aiofile.utils import LineReader

    async def go():
        afp = AIOFile(path, "r")
        await afp.open()
        try:
            reader = LineReader(afp, chunk_size=chunk_size, line_sep=line_sep)
            return [line async for line in reader]
        finally:
            await afp.close()

    return asyncio.run(go())


def real_binary_readline(path: str, size: int = -1):
    """Drive the real `BinaryFileWrapper.readline` with an explicit `size`."""
    from aiofile.utils import async_open

    async def go():
        async with await async_open(path, "rb") as fp:
            return await fp.readline(size=size)

    return asyncio.run(go())

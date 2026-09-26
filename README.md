# mojo-aiofile

`mojo-aiofile` is a Mojo port of the arithmetic inside
[aiofile](https://aiofile.readthedocs.io/) 3.12 — its mode parser and its
chunked-read bookkeeping.

**aiofile is an `io_uring` I/O wrapper.** `AIOFile`, the `caio` context store,
the executor plumbing, `BinaryFileWrapper` and `TextFileWrapper` are all I/O
and awaitables, and none of them is ported. Two things in aiofile are arithmetic
rather than I/O, and they run on every open and every chunk of every read:

- `aiofile.aio.parse_mode`, which turns a mode string into the `os.O_*` flag
  word. That is bit arithmetic and a small state machine over the mode
  characters, and it decides how the file is opened.
- the chunk/offset bookkeeping in `aiofile.utils`: the multibyte decode retry
  schedule in `unicode_reader`, the running read offset `Reader` and `Writer`
  advance, the separator scan `LineReader` performs, and the `size` clamp in
  `BinaryFileWrapper.readline`.

The Python package is `mojo_aiofile`, so it installs alongside the real
`aiofile` and the parity tests compare the two directly.

## Covered subset

| area | upstream source | ported API |
| --- | --- | --- |
| Mode parsing | `aiofile/aio.py` `parse_mode` | `parse_mode` |
| Decode retry schedule | `aiofile/utils.py` `unicode_reader` | `decode_retry_schedule`, `decode_retry_lengths` |
| Running read offset | `aiofile/utils.py` `Reader.read_chunk`, `Writer.__call__` | `running_offsets` |
| Separator scan | `aiofile/utils.py` `LineReader.readline` | `find_sep` |
| `readline` size clamp | `aiofile/utils.py` `BinaryFileWrapper.readline` | `readline_clamp` |

All of it is integer and byte arithmetic, so the parity tests use exact
equality, and every `os.O_*` constant is passed into the kernel from the running
platform rather than hard-coded, so the flag word is exact wherever this is
built.

**Not implemented, and not attempted:** `AIOFile` and every `caio`/io_uring
call, the executor plumbing, `FileIOCloner`, the actual reads and writes, and
the `os.O_*` semantics themselves. Use the real `aiofile` for I/O. Note also
that aiofile's `parse_mode` is deliberately lenient — it ignores unknown
characters and accepts a repeated `x` — and this port reproduces that leniency
rather than tightening it, with tests that pin the behaviour.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aiofile.so
PYTHONPATH=python python -m pytest tests -q
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo = "==1.2.0.dev2026092605"`). Do not run `pixi install` in this tree; the
shared environment at `/nvme0n1-disk/mojo-toolchain` is the environment.

```python
import mojo_aiofile as maf

maf.parse_mode("rb+").flags          # the os.O_* word aiofile would open with
maf.decode_retry_schedule(100)       # [100, 101, 102, 103] for utf-8
maf.running_offsets([10, 20, 30])    # ([10, 30, 60], 60)
maf.find_sep(b"a\nb", b"\n")         # 1
```

## Performance

Best-of-seven wall clock, same process, every case checked against the real
`aiofile` function before timing.

| case | reference | mojo-aiofile | result |
| --- | ---: | ---: | ---: |
| `parse_mode`, one call each, n=65536 | 206.72 ms | 1531.78 ms | **0.13x, a slowdown** |
| `parse_mode`, one long mode string, n=65536 bytes | 22.48 ms | 0.75 ms | 29.80x faster |
| `running_offsets` n=262144 | 77.66 ms | 28.05 ms | 2.77x faster |
| `find_sep` over 4 MiB | 0.15 ms | 3.18 ms | **0.05x, a slowdown** |
| `find_sep` once per 64-byte line, 4 MiB | 56.86 ms | 939.11 ms | **0.06x, a slowdown** |

`parse_mode` on real mode strings loses badly, and it is worth being blunt about
why: a real mode is one to four characters. There is no loop to speed up, so
all the per-call row measures is the cost of a ctypes transition, about 1.5 us,
against a Python function that does the same work in a fraction of that. The
long-mode row is the same kernel with the transition amortised over 65536
characters, and there the compiled loop is 30x faster — which is the honest
measure of the loop itself, and also a reminder that no real caller ever
supplies a mode that long.

`find_sep` loses to `bytes.find` for the same reason and one more: CPython's
`find` is a two-way algorithm with word-at-a-time scanning, and a byte-at-a-time
compiled loop cannot beat it. The port is still the right call for a caller
holding a buffer that is not a Python `bytes` object — a memoryview over an
mmap, say — but against `bytes.find` it is a loss, and it stays one.

`running_offsets` is the case where the arithmetic is actually array-shaped and
the compiled version wins.

Reproduce with:

```bash
python bench/bench.py
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-aiofile.so`.

The `python/mojo_aiofile` layer owns every array. Buffers cross the C ABI as
64-bit addresses (`ctypes.c_int64`; `c_int` truncates them and segfaults) and
are reconstructed in Mojo as pointers, which keeps the exported symbols
non-parametric. The eight `os.O_*` values travel as ordinary integer arguments,
read from `os` at import time.

`parse_mode` returns a small `FileMode` object with the same field order and
comparison behaviour as `aiofile.aio.FileMode`, so a parity test can compare
the two directly.

## Tests

103 parity tests against the real `aiofile`:

- `parse_mode` against `aiofile.aio.parse_mode` over 40 mode strings, covering
  every access character, every `+`/`b` combination, duplicates and junk; plus
  the flag word checked against `os.O_ACCMODE`, the individual `os.O_CREAT` /
  `O_TRUNC` / `O_APPEND` / `O_EXCL` bits, and the leniency (unknown characters
  ignored, `xx` accepted, `""` falling through to `O_WRONLY`);
- the decode retry schedule against `aiofile.utils.ENCODING_MAP` and, end to
  end, against the real `unicode_reader` fed a UTF-8 payload whose chunk
  boundary splits a two-byte character — the reader grows the read by exactly
  one byte, which is what the schedule says it will;
- `running_offsets` against the real `aiofile.utils.Reader` over a real file at
  a 64-byte chunk size, and against the file size in total;
- `find_sep` against `bytes.find` over 200 random buffer/separator pairs, and
  against the real `LineReader` on a file with a 9000-character line and a last
  line with no terminator;
- `readline_clamp` at twelve (buffered, size) pairs, and against the real
  `BinaryFileWrapper.readline` at seven sizes, including the inclusive boundary
  at `size == buffered`.

## License

MIT

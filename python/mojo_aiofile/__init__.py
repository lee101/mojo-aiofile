"""mojo-aiofile: aiofile's mode parsing and chunked-read arithmetic, in Mojo.

aiofile is an `io_uring`-backed async file wrapper: `AIOFile`, the executor
plumbing, `BinaryFileWrapper` and `TextFileWrapper` are I/O and awaitables, and
none of them is ported. What is arithmetic inside aiofile, and runs on every
open and every chunk of every read, is `aiofile.aio.parse_mode` -- the mode
string to `os.O_*` flag word conversion -- and the chunk/offset bookkeeping in
`aiofile.utils` that decides how many bytes to ask for and where the next read
starts. That is what this package implements.

Installable alongside the real `aiofile`, which the parity tests compare
against directly.
"""

from ._lib import (
    DEFAULT_ENCODING_MAX_LEN,
    ENCODING_MAX_LEN,
    OS_FLAGS,
    BadMode,
    FileMode,
    decode_retry_lengths,
    decode_retry_schedule,
    find_sep,
    parse_mode,
    readline_clamp,
    running_offsets,
)

__all__ = [
    "decode_retry_lengths",
    "decode_retry_schedule",
    "find_sep",
    "parse_mode",
    "readline_clamp",
    "running_offsets",
    "BadMode",
    "FileMode",
    "ENCODING_MAX_LEN",
    "DEFAULT_ENCODING_MAX_LEN",
    "OS_FLAGS",
]
__version__ = "0.1.0"

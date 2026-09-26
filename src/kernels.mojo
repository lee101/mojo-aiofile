"""Compiled arithmetic for aiofile's mode parsing and chunked read planning.

aiofile is a thread-pool wrapper over POSIX file I/O built on `caio`/io_uring.
Almost all of it is I/O and awaitables: `AIOFile`, the executor plumbing, the
`BinaryFileWrapper`/`TextFileWrapper` methods. Two things in it are arithmetic
rather than I/O, and they run on every open and every chunk of every read:

  * `aiofile.aio.parse_mode`, which turns a mode string into the `os.O_*` flag
    word, validating as it goes. That is bit arithmetic and a state machine over
    the mode characters, and it decides how the file is opened.
  * the chunk/offset arithmetic of `aiofile.utils`: the multibyte decode retry
    schedule in `unicode_reader`, the running read offset that `Reader` and
    `Writer` advance, the separator scan `LineReader` does over its buffer, and
    the `size` clamp in `BinaryFileWrapper.readline`.

This compilation unit implements those. Every `os.O_*` constant is passed in
rather than hard-coded, so the port is exact on any platform's flag values.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""


def uptr(addr: Int) -> Pointer[UInt8, AnyOrigin[mut=True]]:
    return Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=addr)


def lptr(addr: Int) -> Pointer[Int64, AnyOrigin[mut=True]]:
    return Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# aiofile.aio.parse_mode
# ---------------------------------------------------------------------------

# out slots: 0 readable, 1 writable, 2 plus, 3 appending, 4 created, 5 flags,
#            6 binary, 7 status. Status 0 is a valid mode, -1 is the "Bad mode"
#            that upstream raises.
#
# The eight flag arguments are os.O_RDONLY, O_WRONLY, O_RDWR, O_EXCL, O_CREAT,
# O_TRUNC, O_APPEND and O_BINARY, in that order.


@export("af_parse_mode")
def af_parse_mode(
    mode_addr: Int,
    n: Int,
    o_rdonly: Int,
    o_wronly: Int,
    o_rdwr: Int,
    o_excl: Int,
    o_creat: Int,
    o_trunc: Int,
    o_append: Int,
    o_binary: Int,
    out_addr: Int,
) abi("C") -> Int:
    """Parse a Python file mode string into the `os.O_*` flag word.

    A faithful transcription of `aiofile.aio.parse_mode`, including the order
    the flags are OR-ed in and the way a second read/write/create character is
    rejected. Returns 0, or -1 for a bad mode.
    """
    var p = uptr(mode_addr)
    var o = lptr(out_addr)
    for k in range(8):
        o[unsafe_offset=k] = 0
    o[unsafe_offset=7] = -1
    var flags = Int64(o_rdonly)
    var rwa = 0
    var writable = 0
    var readable = 0
    var plus = 0
    var appending = 0
    var created = 0
    var binary = 0
    for k in range(n):
        var m = p[unsafe_offset=k]
        if m == 120:  # 'x'
            rwa = 1
            created = 1
            writable = 1
            flags = flags | Int64(o_excl) | Int64(o_creat)
        if m == 114:  # 'r'
            if rwa == 1:
                return -1
            rwa = 1
            readable = 1
        if m == 119:  # 'w'
            if rwa == 1:
                return -1
            rwa = 1
            writable = 1
            flags = flags | Int64(o_creat) | Int64(o_trunc)
        if m == 97:  # 'a'
            if rwa == 1:
                return -1
            rwa = 1
            writable = 1
            appending = 1
            flags = flags | Int64(o_creat) | Int64(o_append)
        if m == 43:  # '+'
            if plus == 1:
                return -1
            readable = 1
            writable = 1
            plus = 1
        if m == 98:  # 'b'
            binary = 1
            flags = flags | Int64(o_binary)
    if readable == 1 and writable == 1:
        flags = flags | Int64(o_rdwr)
    elif readable == 1:
        flags = flags | Int64(o_rdonly)
    else:
        flags = flags | Int64(o_wronly)
    o[unsafe_offset=0] = Int64(readable)
    o[unsafe_offset=1] = Int64(writable)
    o[unsafe_offset=2] = Int64(plus)
    o[unsafe_offset=3] = Int64(appending)
    o[unsafe_offset=4] = Int64(created)
    o[unsafe_offset=5] = flags
    o[unsafe_offset=6] = Int64(binary)
    o[unsafe_offset=7] = 0
    return 0


# ---------------------------------------------------------------------------
# aiofile.utils.unicode_reader retry schedule
# ---------------------------------------------------------------------------


@export("af_decode_retry_schedule")
def af_decode_retry_schedule(chunk_size: Int, max_len: Int, out_addr: Int) abi("C") -> Int:
    """The read lengths `unicode_reader` tries, in order.

    `for retry in range(ENCODING_MAP.get(encoding, 4)): read(chunk_size +
    retry)` -- the extra bytes cover a multibyte character split across the
    chunk boundary. Writes the lengths to `out` and returns how many there are.
    A negative `chunk_size` short-circuits into a single read of -1, as
    upstream does.
    """
    var o = lptr(out_addr)
    if chunk_size < 0:
        o[unsafe_offset=0] = -1
        return 1
    var count = max_len
    if count < 1:
        count = 1
    for i in range(count):
        o[unsafe_offset=i] = Int64(chunk_size + i)
    return count


@export("af_decode_retry_lengths")
def af_decode_retry_lengths(
    chunk_size: Int, max_len: Int, attempts: Int, out_addr: Int
) abi("C") -> Int:
    """The first `attempts` lengths of the retry schedule.

    `unicode_reader` returns after the first decode that succeeds, so the length
    actually read is `chunk_size + k` for the smallest `k` that decoded. This
    is what a caller planning a read needs. Returns how many were written.
    """
    var o = lptr(out_addr)
    if attempts <= 0:
        return 0
    var count = attempts
    if count > max_len:
        count = max_len
    for i in range(count):
        o[unsafe_offset=i] = Int64(chunk_size + i)
    return count


# ---------------------------------------------------------------------------
# aiofile.utils.Reader / Writer offset arithmetic
# ---------------------------------------------------------------------------


@export("af_running_offsets")
def af_running_offsets(
    sizes_addr: Int, n: Int, start: Int, out_addr: Int
) abi("C") -> Int:
    """The offset after each read or write, as `Reader.read_chunk` advances it.

    `self.__offset += len(chunk)`, and the chunk length is whatever the read
    returned. `sizes` holds the realised chunk lengths; `out` receives the
    running offset after each one, starting from `start`. Returns the final
    offset.
    """
    var sizes = lptr(sizes_addr)
    var o = lptr(out_addr)
    var offset = Int64(start)
    for i in range(n):
        offset = offset + sizes[unsafe_offset=i]
        o[unsafe_offset=i] = offset
    return Int(offset)


# ---------------------------------------------------------------------------
# aiofile.utils.LineReader separator scan
# ---------------------------------------------------------------------------


@export("af_find_sep")
def af_find_sep(
    buf_addr: Int, n: Int, sep_addr: Int, sep_len: Int, start: Int
) abi("C") -> Int:
    """Index of the first occurrence of a `sep_len` byte separator, or -1.

    `LineReader.readline` asks its buffer for a line with `readline`, which is
    CPython's own scan for the separator; this is the same search over a raw
    buffer, for a caller holding `bytes` rather than a `BytesIO`.
    """
    if sep_len <= 0 or start < 0 or start > n:
        return -1
    var p = uptr(buf_addr)
    var s = uptr(sep_addr)
    if sep_len == 1:
        var needle = s[unsafe_offset=0]
        var i = start
        while i < n:
            if p[unsafe_offset=i] == needle:
                return i
            i += 1
        return -1
    var last = n - sep_len
    var i = start
    while i <= last:
        var j = 0
        while j < sep_len and p[unsafe_offset=i + j] == s[unsafe_offset=j]:
            j += 1
        if j == sep_len:
            return i
        i += 1
    return -1


# ---------------------------------------------------------------------------
# aiofile.utils readline size clamp
# ---------------------------------------------------------------------------


@export("af_readline_clamp")
def af_readline_clamp(buffered: Int, size: Int, out_addr: Int) abi("C") -> Int:
    """The `0 < size <= fp.tell()` branch of `BinaryFileWrapper.readline`.

    Upstream seeks to `size` and truncates there when the buffer already holds
    at least `size` bytes; otherwise it returns the first line from the buffer.
    Writes the resulting length to `out` and returns 1 when the clamp applies,
    0 when it does not.
    """
    var o = lptr(out_addr)
    var have = Int64(buffered)
    if size > 0 and Int64(size) <= have:
        o[unsafe_offset=0] = Int64(size)
        return 1
    o[unsafe_offset=0] = have
    return 0

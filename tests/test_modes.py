"""Parity against `aiofile.aio.parse_mode`.

Mode parsing is bit arithmetic over the `os.O_*` flags and a state machine
over the mode characters, and it is exact: a single wrong OR or a missing
validation gives a different flag word or accepts a mode upstream rejects.
"""

import os

import pytest

import mojo_aiofile as maf
from conftest import real_parse_mode

MODES = [
    "r", "w", "a", "x",
    "rb", "wb", "ab", "xb", "br", "bw", "bx", "ba",
    "r+", "w+", "a+", "x+", "+r", "+w", "+a", "+x",
    "rb+", "r+b", "wb+", "ab+", "xb+", "b+r",
    "rbb", "www", "rrr", "tt", "rt", "rbt", "r+t",
    "rbt+", "brt+",
    "", "z", "rw", "ra", "wa", "ax", "xx", "r++", "rbb+",
    "br+", "bw+", "ba+", "bx+", "tr", "ub", "rr+",
]


@pytest.mark.parametrize("mode", MODES)
def test_parse_mode_matches_aiofile(mode):
    theirs, exc = real_parse_mode(mode)
    if exc is not None:
        with pytest.raises(maf.BadMode):
            maf.parse_mode(mode)
        assert "Bad mode" in str(exc)
    else:
        assert maf.parse_mode(mode).as_tuple() == theirs


@pytest.mark.parametrize("mode", ["r", "w", "a", "x", "r+", "rb", "ab", "r+b"])
def test_parse_mode_flags_match_the_os_flags(mode):
    """The flag word is what actually opens the file, so check it against the
    values the kernel call would use rather than only against aiofile."""
    theirs, _exc = real_parse_mode(mode)
    assert maf.parse_mode(mode).flags == theirs.flags
    # readable-only must not be writable and vice versa
    m = maf.parse_mode(mode)
    if m.readable and not m.writable:
        assert m.flags & os.O_ACCMODE == os.O_RDONLY
    elif m.writable and not m.readable:
        assert m.flags & os.O_ACCMODE == os.O_WRONLY
    else:
        assert m.flags & os.O_ACCMODE == os.O_RDWR


def test_parse_mode_sets_the_creation_flags():
    assert maf.parse_mode("w").flags & os.O_CREAT
    assert maf.parse_mode("w").flags & os.O_TRUNC
    assert maf.parse_mode("a").flags & os.O_APPEND
    assert maf.parse_mode("x").flags & os.O_EXCL
    assert not maf.parse_mode("r").flags & os.O_CREAT


def test_parse_mode_x_creates_but_r_never_does():
    assert maf.parse_mode("x").created is True
    assert maf.parse_mode("w").created is False
    assert maf.parse_mode("r").created is False


def test_parse_mode_binary_only_affects_the_flag():
    text = maf.parse_mode("r")
    binary = maf.parse_mode("rb")
    assert text.binary is False and binary.binary is True
    assert binary.flags == text.flags | getattr(os, "O_BINARY", 0)


def test_parse_mode_rejects_a_second_access_character():
    for mode in ("rw", "rr", "ra", "wa", "wr", "xr", "r++", "aa", "wr+"):
        with pytest.raises(maf.BadMode):
            maf.parse_mode(mode)


def test_parse_mode_ignores_unknown_characters():
    """aiofile's `parse_mode` only reacts to the characters it knows; anything
    else is silently ignored rather than rejected. A port that validated would
    disagree with upstream, so pin the leniency."""
    for mode in ("z", "rz", "q+", "rbtq", "r\u00e9"):
        theirs, exc = real_parse_mode(mode)
        assert exc is None
        assert maf.parse_mode(mode).as_tuple() == theirs
    assert maf.parse_mode("r").flags == maf.parse_mode("rz").flags


def test_parse_mode_rejects_a_second_create_character():
    """`x` sets the read/write/append guard but does not check it, so `xx` is
    accepted upstream. A port that tightened this would break real code."""
    theirs, exc = real_parse_mode("xx")
    assert exc is None
    assert maf.parse_mode("xx").as_tuple() == theirs


def test_parse_mode_empty_string_is_write_only():
    """No characters at all means neither readable nor writable, and upstream's
    final `else` falls through to O_WRONLY. Pin that rather than treating it
    as an error."""
    theirs, exc = real_parse_mode("")
    assert exc is None
    m = maf.parse_mode("")
    assert m.as_tuple() == theirs
    assert m.readable is False and m.writable is False
    assert m.flags & os.O_ACCMODE == os.O_WRONLY

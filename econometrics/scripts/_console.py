"""Console output that cannot stop an analysis on Windows.

The modules print summaries with characters such as an arrow or a less-or-equal sign. On Windows,
when output is captured (Claude Code running a delivered script, a pipe, a log file), Python
encodes stdout with the ANSI code page, usually cp1252, which has no arrow, and print() raises
UnicodeEncodeError in the middle of the analysis.

Importing this module checks stdout and stderr once. When a stream's error handler raises on such
characters ("strict", or "surrogateescape", which a PowerShell pipe gives) and its encoding cannot
write them, its error handler becomes "replace": an unencodable character
prints as "?" and the run goes on. The encoding and every other character are unchanged. Nothing
happens on UTF-8 streams (Linux, macOS, Windows with PYTHONUTF8=1), on streams that are already
tolerant (stderr uses "backslashreplace" by default) or on streams without reconfigure (StringIO,
notebooks).

When stdout or stderr is a regular file (a log written with `py script.py > log.txt`) and its encoding is not
UTF-8, it is switched to UTF-8 instead (v2.4.6.2): a file has no console to match, and the cp1252 default turned
arrows into "?" and accented words into mojibake in a log read in an editor; the agent in the IPCA validation
even removed the accents from its Portuguese output. Pipes and consoles are left as described above, because the
program reading them decodes with the console code page. This covers `>` in cmd and in Git Bash (the shell
Claude Code uses on Windows); PowerShell redirects through a pipe and re-encodes the output itself, so there the
behaviour is the one for pipes. Import _console before printing anything: text printed earlier, or a log
appended with `>>` to an older cp1252 file, would leave two encodings in one file.

Modules that print such characters import it right after their own imports:

    import _console  # noqa: F401  (Windows-safe console output)

tests/test_windows_console.py fails if a module prints such a character without that import.
"""
from __future__ import annotations

import os
import stat
import sys

# Characters the modules print, plus close relatives; a stream that cannot write one of them
# gets the tolerant handler.
_PROBE = "\u2192\u2190\u2264\u2265\u2260\u2248\xb1\u221a\u03b1\u03b2\u03c3\u03c1\u2713\u2717\u2014"

# Error handlers that already write something for a character the encoding lacks.
_TOLERANT = {"replace", "backslashreplace", "ignore", "xmlcharrefreplace", "namereplace"}


def _is_regular_file(stream) -> bool:
    try:  # io.UnsupportedOperation (StringIO, BytesIO-backed wrappers) subclasses OSError and ValueError
        return stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
    except (AttributeError, OSError, ValueError):
        return False


def make_output_tolerant() -> None:
    """Switch stdout/stderr streams that would raise on _PROBE to errors="replace"; write regular files in UTF-8."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        encoding = getattr(stream, "encoding", None)
        reconfigure = getattr(stream, "reconfigure", None)
        if (encoding and reconfigure is not None and encoding.replace("-", "").lower() not in ("utf8", "utf8sig")
                and _is_regular_file(stream)):
            try:
                # keep the stream's error handler (backslashreplace on stderr, or what PYTHONIOENCODING chose)
                reconfigure(encoding="utf-8", errors=getattr(stream, "errors", None) or "strict")
                continue
            except (ValueError, OSError):  # data already read/written in a way that forbids it: fall through
                pass
        # "strict" and "surrogateescape" (what a PowerShell pipe gives) both raise on an arrow
        if not encoding or reconfigure is None or getattr(stream, "errors", None) in _TOLERANT:
            continue
        try:
            _PROBE.encode(encoding)
        except UnicodeEncodeError:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # closed or detached stream: leave it as it is
                pass
        except LookupError:  # unknown encoding name: leave it as it is
            pass


make_output_tolerant()

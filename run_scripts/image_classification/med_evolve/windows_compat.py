"""Provide the small POSIX file-lock API used by GigaEvo during Windows smoke tests."""

import errno
import os
import sys
import time
import types


def install_windows_flock() -> None:
    """Use real Windows byte-range locks; leave Ubuntu's native fcntl untouched."""
    if os.name != "nt" or "fcntl" in sys.modules:
        return
    import msvcrt

    module = types.ModuleType("fcntl")
    module.LOCK_SH, module.LOCK_EX, module.LOCK_NB, module.LOCK_UN = 1, 2, 4, 8

    def flock(fd: int, operation: int) -> None:
        """Serialize file access; shared requests use a conservative exclusive lock."""
        position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation & module.LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if operation & module.LOCK_NB:
                        raise BlockingIOError(errno.EAGAIN, "The file is locked") from None
                    time.sleep(0.05)
        finally:
            os.lseek(fd, position, os.SEEK_SET)

    module.flock = flock
    sys.modules["fcntl"] = module

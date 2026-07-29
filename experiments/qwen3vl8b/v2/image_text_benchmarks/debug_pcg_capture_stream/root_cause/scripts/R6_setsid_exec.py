#!/usr/bin/env python3
"""setsid + exec helper for the R6.1 runner.

Wraps a child command so that:

  * The final process becomes a new session leader
    (PID == PGID == SID).
  * The final process's PID is written to the given pidfile BEFORE
    exec, so the parent runner can read it deterministically and use
    it for scoped signalling ("kill -TERM -PID" sends SIGTERM to the
    whole process group, catching all sglang server-side children).
  * No `pkill` / `killall` / broad process kill is ever needed — the
    parent runner only ever signals PGIDs it recorded here.

Usage:
    R6_setsid_exec.py <pidfile> <cmd> [args...]

Exit codes:
    64 usage error (bad args)
    <other> whatever the exec'd process exits with (via exec, so this
    wrapper does not linger; its PID becomes the child's PID)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: R6_setsid_exec.py <pidfile> <cmd> [args...]",
              file=sys.stderr)
        return 64
    pidfile = sys.argv[1]
    cmd = sys.argv[2:]
    # Best-effort setsid — may already be a session leader in unusual
    # invocation environments, in which case os.setsid raises OSError
    # and we proceed with the existing session.
    try:
        os.setsid()
    except OSError:
        pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    os.execvp(cmd[0], cmd)
    # unreachable
    return 127


if __name__ == "__main__":
    sys.exit(main())

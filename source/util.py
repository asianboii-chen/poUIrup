import datetime
import os
import signal
import sys
import tempfile
import time
import typing


def get_timestamp_for_logging() -> str:
    return datetime.datetime.now().time().isoformat("milliseconds")


def log(*tokens: object, separator: str = " ", file=sys.stdout) -> None:
    print(get_timestamp_for_logging(), end=" ", file=file)
    print(*tokens, sep=separator, file=file)


def log_error(*tokens: object, separator: str = " ") -> None:
    log(*tokens, separator=separator, file=sys.stderr)


def ensure_single_instance(
    handle_new_instance_started: typing.Callable[[], None]
) -> None:
    LOCK_PATH = os.path.join(tempfile.gettempdir(), "pouirup.lock")

    def terminate_running_instance_if_exists() -> None:
        if not os.path.exists(LOCK_PATH):
            return

        with open(LOCK_PATH, "r") as lock:
            try:
                running_pid = int(lock.readline())
            except ValueError:
                return

        try:
            os.kill(running_pid, signal.SIGTERM)
        except OSError:
            return

        wait_seconds = 0.125
        while wait_seconds <= 2:
            time.sleep(wait_seconds)
            wait_seconds *= 2
            try:
                os.kill(running_pid, 0)
            except OSError:
                return

        raise Exception(
            f"The existing process (PID {running_pid}) took too long to exit."
        )

    terminate_running_instance_if_exists()

    with open(LOCK_PATH, "w") as lock:
        lock.writelines([str(os.getpid())])

    signal.signal(signal.SIGTERM, lambda *args: handle_new_instance_started())

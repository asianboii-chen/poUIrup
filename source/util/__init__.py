import os
import os.path
import signal
import tempfile
import time
import typing


def ensure_single_instance(
        on_new_instance_start: typing.Callable[[], None]) -> None:
    LOCK_PATH = os.path.join(tempfile.gettempdir(), f'pouirup.lock')

    def terminate_existing_instance():
        if not os.path.exists(LOCK_PATH):
            return
        with open(LOCK_PATH, 'r') as lock:
            try:
                running_pid = int(lock.readline())
            except ValueError:
                return
        try:
            os.kill(running_pid, signal.SIGTERM)
        except OSError:
            return
        wait_secs = .125
        while wait_secs <= 2:
            time.sleep(wait_secs)
            wait_secs *= 2
            try:
                os.kill(running_pid, 0)
            except OSError:
                return
        raise Exception(
            f'The existing process (PID {running_pid}) took too long to exit.')

    terminate_existing_instance()

    with open(LOCK_PATH, 'w') as lock:
        lock.writelines((str(os.getpid()), ))

    def handle_sigterm(_sig: int, _frame: typing.Any) -> None:
        nonlocal on_new_instance_start
        on_new_instance_start()

    signal.signal(signal.SIGTERM, handle_sigterm)

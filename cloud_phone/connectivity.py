"""Cheap "are we online?" check, so the phone can say it's offline instead of going silent."""

import socket
import threading

from . import config


def is_online() -> bool:
    """True if a TCP connection to ONLINE_CHECK_HOST:443 succeeds within ONLINE_CHECK_TIMEOUT.

    Runs in a thread with a hard deadline because DNS lookups ignore socket
    timeouts and can hang for a long time on wifi without a working uplink.
    """
    result = {"online": False}

    def check():
        try:
            with socket.create_connection((config.ONLINE_CHECK_HOST, 443), timeout=config.ONLINE_CHECK_TIMEOUT):
                result["online"] = True
        except OSError:
            pass

    thread = threading.Thread(target=check, daemon=True)
    thread.start()
    thread.join(config.ONLINE_CHECK_TIMEOUT)
    return result["online"]

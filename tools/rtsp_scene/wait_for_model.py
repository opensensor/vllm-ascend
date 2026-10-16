"""Wait for the scene API before systemd starts camera processing."""

import argparse
import math
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_INTERVAL_SECONDS = 2


def probe_health(endpoint, timeout):
    with urllib.request.urlopen(endpoint, timeout=timeout) as response:
        return response.status == 200


def wait_for_health(
    endpoint,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    interval=DEFAULT_INTERVAL_SECONDS,
    *,
    probe=probe_health,
    clock=time.monotonic,
    sleep=time.sleep,
):
    if not endpoint or not all(math.isfinite(value) and value > 0 for value in (timeout, interval)):
        raise ValueError("Use an endpoint and finite positive timeout/interval values")
    deadline = clock() + timeout
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f"Scene API did not become ready within {timeout} seconds")
        try:
            if probe(endpoint, min(interval, remaining)):
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        sleep(min(interval, max(0, deadline - clock())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003/health")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    args = parser.parse_args()
    wait_for_health(args.endpoint, args.timeout, args.interval)
    print("Scene API ready", flush=True)


if __name__ == "__main__":
    main()

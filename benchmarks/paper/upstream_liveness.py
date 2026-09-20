"""Stop an owned benchmark when its inference service has disappeared."""

import errno
import socket
from urllib.parse import urlsplit


class UpstreamUnavailable(RuntimeError):
    """Infrastructure interruption, never a scored model failure."""


class UpstreamLiveness:
    """Check process ownership and TCP refusal without generating health tokens.

    A busy service can time out while remaining alive. Only repeated connection
    refusal or an observed owned-process exit stops the benchmark. HTTP method
    failures (including context overflow) do not enter this classification.
    """

    def __init__(self, url, *, process=None, refused_limit=3, timeout=1.0):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("An HTTP inference URL is required")
        self.address = (parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        self.process = process
        self.refused_limit = refused_limit
        self.timeout = timeout
        self.refused = 0

    def __call__(self):
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                raise UpstreamUnavailable(
                    f"Inference service exited with code {code}; stop the current cell")
        try:
            with socket.create_connection(self.address, timeout=self.timeout):
                pass
        except OSError as error:
            if error.errno not in {errno.ECONNREFUSED, 10061}:
                # Timeouts and temporary network failures are request-level
                # infrastructure errors, not proof that the server has exited.
                self.refused = 0
                return
            self.refused += 1
            if self.refused >= self.refused_limit:
                raise UpstreamUnavailable(
                    f"Inference service {self.address[0]}:{self.address[1]} refused "
                    f"{self.refused} consecutive connections; stop the current cell") from error
        else:
            self.refused = 0

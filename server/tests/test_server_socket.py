import http.client
import socket
import threading
import unittest

import _path  # noqa: F401
import server


class Probe(server.Handler):
    """The real handler, with a route that reports how its socket is set up."""

    seen: list = []

    def do_GET(self):
        Probe.seen.append(self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
        self._json({"ok": True})


class SpeakerSockets(unittest.TestCase):
    def test_small_writes_to_the_speaker_go_out_at_once(self):
        """With Nagle's algorithm a small write waits until the previous one has been acknowledged, and the
        speaker acknowledges late (up to a quarter of a second): that wait could land on every chunk of a
        streamed reply. Python's http.server leaves Nagle on unless told otherwise."""
        Probe.seen.clear()
        httpd = server.Server(("127.0.0.1", 0), Probe)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/probe")
        self.assertEqual(conn.getresponse().status, 200)
        self.assertEqual(len(Probe.seen), 1)
        self.assertTrue(Probe.seen[0], "TCP_NODELAY is off on the connection to the speaker")


if __name__ == "__main__":
    unittest.main()

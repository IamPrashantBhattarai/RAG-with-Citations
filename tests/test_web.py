import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from citerag import config, layout, web

PAGE = b"<html><head><title>T</title></head><body><nav>Home Shop</nav><h1>Solar guide</h1><p>A 3 kW system costs NPR 330,000.</p></body></html>"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_address[1]
        routes = {
            "/page.html": (200, "text/html; charset=utf-8", PAGE),
            "/doc.pdf": (200, "application/pdf", b"%PDF-1.4 fake"),
            "/image": (200, "image/png", b"\x89PNG"),
            "/missing": (404, "text/html", b"no"),
        }
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/page.html")  # not an allowed host
            self.end_headers()
            return
        if self.path == "/hop":
            self.send_response(301)
            self.send_header("Location", "/page.html")
            self.end_headers()
            return
        status, kind, body = routes[self.path]
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def only_127(host):
    return host == "127.0.0.1"  # the test server, standing in for a public site


def test_page_is_saved_with_its_address_and_read_as_a_document(server, tmp_path, monkeypatch):
    path = web.fetch(f"{server}/hop", tmp_path, allowed=only_127)  # follows a same-site redirect
    assert path.name == "127.0.0.1-page.html"
    assert path.read_bytes().startswith(b'<meta name="citerag-source-url" content="' + server.encode())
    monkeypatch.setattr(config, "PARSE_CACHE_DIR", tmp_path / "parsed")
    if layout.available():
        [text] = layout.parse_file(path)
        assert text.startswith(f"Web page: {server}/page.html\n\n# Solar guide") and "Home Shop" not in text
    assert web.fetch(f"{server}/doc.pdf", tmp_path, allowed=only_127).name == "127.0.0.1-doc.pdf"


def test_refused_pages_say_why(server, tmp_path, monkeypatch):
    with pytest.raises(web.WebError, match="localhost is not a public internet address"):
        web.fetch(f"{server}/redirect", tmp_path, allowed=only_127)  # checked again after a redirect
    with pytest.raises(web.WebError, match="image/png isn't a web page or PDF"):
        web.fetch(f"{server}/image", tmp_path, allowed=only_127)
    with pytest.raises(web.WebError, match="answered 404"):
        web.fetch(f"{server}/missing", tmp_path, allowed=only_127)
    with pytest.raises(web.WebError, match="only http:// and https://"):
        web.fetch("ftp://example.org/file", tmp_path, allowed=only_127)
    with pytest.raises(web.WebError, match="127.0.0.1 is not a public"):
        web.fetch(f"{server}/page.html", tmp_path)  # the real check: loopback is refused
    monkeypatch.setattr(web, "MAX_BYTES", 10)
    with pytest.raises(web.WebError, match="larger than"):
        web.fetch(f"{server}/page.html", tmp_path, allowed=only_127)
    assert list(tmp_path.iterdir()) == []  # nothing half-saved


def test_only_public_addresses_are_allowed(monkeypatch):
    def resolve(*ips):
        monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", (ip, 0)) for ip in ips])

    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.20", "169.254.169.254", "::1", "fd00::1", "100.64.0.1"):
        resolve(ip)
        assert not web.public_host("example.org"), ip
    resolve("93.184.216.34")
    assert web.public_host("example.org")
    resolve("93.184.216.34", "10.0.0.5")  # one private address is enough to refuse
    assert not web.public_host("example.org")

    def fail(host, port):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert not web.public_host("no-such-host.invalid")


def test_file_names_come_from_the_address():
    assert web.file_name("https://www.example.org/blog/solar-guide.html?x=1", ".html") == "example.org-blog-solar-guide.html"
    assert web.file_name("https://example.org/", ".html") == "example.org.html"
    assert web.file_name("https://example.org/a b/ç", ".pdf") == "example.org-a-b.pdf"
    assert len(web.file_name("https://example.org/" + "x" * 300, ".html")) == 105

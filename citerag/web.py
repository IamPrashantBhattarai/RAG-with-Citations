"""Save a web page (or a PDF on the web) into the documents folder. (Phase 11)

The page is downloaded once and saved as a file, so it is indexed, cited and
re-scanned like any other document, and answers keep working if the page
changes or disappears. An HTML page gets a meta tag with its address, which
the loader puts at the top of its text ("Web page: https://…").

The app runs on the user's computer, so a URL must not be able to make it read
the local network: only http(s), and every address the host name resolves to
must be public (no localhost, 10.x, 192.168.x, link-local cloud metadata…),
checked again at every redirect. Also capped: 5 redirects, 15 s, 5 MB, and
only HTML or PDF responses.

Try it:  python -m citerag.web https://example.org/page   (then index data/)
"""

from __future__ import annotations

import argparse
import datetime
import html
import ipaddress
import re
import socket
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx

from citerag import config

MAX_BYTES = 5 * 1024 * 1024
TIMEOUT = 15.0
MAX_REDIRECTS = 5
USER_AGENT = "CiteRAG/1.0 (document Q&A; saves one page on request)"
TYPES = {"text/html": ".html", "application/xhtml+xml": ".html", "application/pdf": ".pdf"}


class WebError(Exception):
    """The page could not be saved; the message says why, for the user."""


def public_host(host: str) -> bool:
    """True when every address `host` resolves to is a public internet address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    addresses = {info[4][0].split("%")[0] for info in infos}
    return bool(addresses) and all(ipaddress.ip_address(a).is_global for a in addresses)


def fetch(url: str, folder: Path = config.DATA_DIR,
          allowed: Callable[[str], bool] = public_host) -> Path:
    """Download `url` into `folder`; returns the saved file. `allowed` vets each host."""
    url = url.strip()
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise WebError(f"{url}: only http:// and https:// addresses can be added")
        if not allowed(parts.hostname):
            raise WebError(f"{url}: {parts.hostname} is not a public internet address")
        try:
            with httpx.stream("GET", url, timeout=TIMEOUT, follow_redirects=False,
                              headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf"}) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code != 200:
                    raise WebError(f"{url}: the site answered {r.status_code} {r.reason_phrase}")
                kind = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind not in TYPES:
                    raise WebError(f"{url}: {kind or 'unknown content'} isn't a web page or PDF")
                if int(r.headers.get("content-length") or 0) > MAX_BYTES:
                    raise WebError(f"{url}: larger than {MAX_BYTES // 2**20} MB")
                body = b""
                for piece in r.iter_bytes():
                    body += piece
                    if len(body) > MAX_BYTES:
                        raise WebError(f"{url}: larger than {MAX_BYTES // 2**20} MB")
        except httpx.HTTPError as e:
            raise WebError(f"{url}: couldn't download it ({type(e).__name__})") from e
        return _save(url, body, TYPES[kind], folder)
    raise WebError(f"{url}: more than {MAX_REDIRECTS} redirects")


def file_name(url: str, suffix: str) -> str:
    """"https://www.example.org/blog/solar-guide?x=1" -> "example.org-blog-solar-guide.html"."""
    parts = urlsplit(url)
    host = (parts.hostname or "page").removeprefix("www.")
    path = re.sub(r"\.(html?|php|aspx?|pdf)$", "", parts.path.strip("/"), flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{host}-{path}" if path else host).strip("-.")
    return f"{stem[:100]}{suffix}"


def _save(url: str, body: bytes, suffix: str, folder: Path) -> Path:
    if suffix == ".html":  # record where it came from, for the loader
        day = datetime.date.today().isoformat()
        meta = (f'<meta name="citerag-source-url" content="{html.escape(url, quote=True)}">'
                f'<meta name="citerag-saved" content="{day}">\n').encode()
        body = meta + body
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name(url, suffix)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Save a web page into the documents folder.")
    parser.add_argument("url")
    parser.add_argument("--folder", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()
    try:
        print(f"Saved {fetch(args.url, args.folder)}. Index it with: python -m citerag.store {args.folder}/")
    except WebError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    _main()

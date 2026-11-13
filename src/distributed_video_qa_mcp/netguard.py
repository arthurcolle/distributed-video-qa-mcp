from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from .errors import VideoQAError


def is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url: str, *, allow_private: bool = False) -> str:
    """Reject anything that is not a plain http(s) URL to a public host.

    The server fetches URLs on behalf of a model, so a link to localhost, a cloud
    metadata address or a private network would otherwise be a way to read things
    the caller could not reach. Resolution happens here and again inside the
    downloader, so this narrows the window rather than closing it against a
    hostile DNS server; run with network egress controls when that matters.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"}:
        raise VideoQAError("bad_url", "only http and https URLs are accepted")
    if not parts.hostname:
        raise VideoQAError("bad_url", "the URL has no host")
    if parts.username or parts.password:
        raise VideoQAError("bad_url", "URLs with embedded credentials are not accepted")
    if allow_private:
        return parts.geturl()
    host = parts.hostname
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise VideoQAError("bad_url", f"could not resolve {host}") from exc
    addresses = {str(info[4][0]).split("%")[0] for info in infos}
    blocked = sorted(a for a in addresses if not _is_public(a))
    if blocked or not addresses:
        raise VideoQAError(
            "private_host",
            f"{host} resolves to a non-public address; set DVQA_ALLOW_PRIVATE_HOSTS=1 to permit it",
        )
    return parts.geturl()

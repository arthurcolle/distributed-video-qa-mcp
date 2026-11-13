from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import logging
import sys
from typing import Any

from . import __version__, retrieval
from .config import Settings, load_settings
from .errors import VideoQAError
from .pipeline import IngestOptions
from .server import App, build_server


def _log(level: str) -> None:
    # stdout carries the MCP protocol on stdio, so every log line goes to stderr.
    logging.basicConfig(level=level, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class BearerAuth:
    """Require `Authorization: Bearer <token>` on everything except /health."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return
        supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
        if hmac.compare_digest(supplied, b"Bearer " + self.token):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"unauthorized"}'
        headers = [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]
        await send({"type": "http.response.start", "status": 401, "headers": headers})
        await send({"type": "http.response.body", "body": body})


async def _serve(args: argparse.Namespace) -> int:
    settings = load_settings(http=args.http)
    mcp, app = build_server(settings)
    try:
        if not args.http:
            await mcp.run_stdio_async()
            return 0
        import uvicorn
        from mcp.server.transport_security import TransportSecuritySettings
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        if not _loopback(args.host) and not settings.http_token:
            raise VideoQAError("config", "set DVQA_HTTP_TOKEN before binding to a non-loopback address")

        @mcp.custom_route("/health", methods=["GET"])
        async def health(_: Request) -> JSONResponse:
            return JSONResponse({"ok": True, "version": __version__, **app.store.stats()})

        security = None
        if args.allowed_host:
            security = TransportSecuritySettings(allowed_hosts=args.allowed_host)
        asgi: Any = mcp.streamable_http_app(host=args.host, transport_security=security)
        if settings.http_token:
            asgi = BearerAuth(asgi, settings.http_token)
        config = uvicorn.Config(asgi, host=args.host, port=args.port, log_level=args.log_level.lower())
        await uvicorn.Server(config).serve()
        return 0
    finally:
        await app.shutdown()


async def _doctor(_: argparse.Namespace) -> int:
    settings = load_settings()
    mcp, app = build_server(settings)
    try:
        result = await mcp.call_tool("video_status", {})
        status = getattr(result, "structured_content", None) or {}
        print(json.dumps(status, indent=2))
        problems = [
            name
            for name in ("ffmpeg", "ffprobe", "transcription", "embedding")
            if str(status.get(name, "")).startswith(("missing", "unavailable"))
        ]
        if problems:
            print(f"\nnot ready: {', '.join(problems)}", file=sys.stderr)
            return 1
        return 0
    finally:
        await app.shutdown()


async def _ingest(args: argparse.Namespace) -> int:
    settings: Settings = load_settings()
    app = App(settings)
    try:
        options = IngestOptions(
            quality=args.quality, language=args.language, frame_interval=args.frame_interval, force=args.force
        )
        job = app.pipeline.start_ingest(args.source, options)
        shown = ""
        while True:
            job = await app.pipeline.wait(job["id"], 1.0)
            line = f"{job['progress'] * 100:5.1f}%  {job['stage'] or 'queued':<10} {job['message']}"
            if line != shown:
                print(line, file=sys.stderr)
                shown = line
            if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
        print(json.dumps(job["result"] or {"status": job["status"], "error": job["error"]}, indent=2))
        return 0 if job["status"] == "succeeded" else 1
    finally:
        await app.shutdown()


async def _search(args: argparse.Namespace) -> int:
    app = App(load_settings())
    try:
        hits = await retrieval.search(app.store, app.embedder, args.query, top_k=args.top_k)
        for hit in hits:
            what = hit.text if hit.kind == "transcript" else f"[frame] {hit.frame_path}"
            print(f"{hit.video_id} {retrieval.clock(hit.start):>8}  {hit.score:.4f}  {what}")
        return 0
    finally:
        await app.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="distributed-video-qa-mcp", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the MCP server (default)")
    serve.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4005)
    serve.add_argument("--allowed-host", action="append", help="Host header to accept; repeat for several")

    sub.add_parser("doctor", help="check tools, backends and the index")

    ingest = sub.add_parser("ingest", help="index a URL or local file from the terminal")
    ingest.add_argument("source")
    ingest.add_argument("--quality")
    ingest.add_argument("--language")
    ingest.add_argument("--frame-interval", type=float)
    ingest.add_argument("--force", action="store_true")

    search = sub.add_parser("search", help="search the index from the terminal")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args([*(argv or sys.argv[1:]), "serve"])
    _log(args.log_level)
    handler = {"serve": _serve, "doctor": _doctor, "ingest": _ingest, "search": _search}[args.command]
    try:
        raise SystemExit(asyncio.run(handler(args)))
    except VideoQAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

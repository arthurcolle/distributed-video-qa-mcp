# Security policy

The 0.x line is maintained. Report a suspected vulnerability through [GitHub private vulnerability reporting](https://github.com/arthurcolle/distributed-video-qa-mcp/security/advisories/new). Include the version, a minimal reproduction using public or synthetic media, and the boundary you expected to hold. Leave out API keys and private recordings.

## Threat model

The server fetches URLs and reads files on behalf of a language model, so its inputs should be treated as untrusted even when the person using it is trusted. The boundaries it enforces:

| Boundary | Enforcement |
| --- | --- |
| Network targets | http(s) only; the host must resolve to public addresses (no loopback, private, link-local, multicast or metadata ranges); no embedded credentials. Opt out with `DVQA_ALLOW_PRIVATE_HOSTS=1`. |
| Filesystem | Paths are resolved, symlinks included, and must lie under `DVQA_ALLOWED_ROOTS` or the state directory. HTTP mode allows no roots by default. |
| Deletion | Only index rows and the server's own media copies are deleted; files ingested by path are never touched. |
| Subprocesses | FFmpeg runs with argument lists and no shell. Format ids may contain only letters, digits and `+-_./`. |
| Resource use | Limits on duration, file size, frames per video, concurrent jobs and batch size; live streams are refused. |
| HTTP transport | Bearer token compared in constant time; non-loopback binding requires a token; optional Host-header allow-list. |
| Secrets | Read from the environment only; never logged or returned by tools. |

## Known limits

- The private-address check resolves the hostname before the download starts, and yt-dlp resolves it again when it connects. A hostile DNS server could answer differently the second time. Where that matters, run the server with network egress controls.
- yt-dlp and FFmpeg parse untrusted media. Keep both updated, and isolate the process (container, separate user) on shared deployments.
- Tool results, including transcripts of whatever was downloaded, are visible to the MCP host and its model.

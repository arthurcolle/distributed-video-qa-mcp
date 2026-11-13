from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError


class VideoQAError(ToolError):
    """An expected failure with a message that is safe to show to the MCP client.

    Subclassing ToolError makes the SDK return the message to the model as an
    error result; any other exception is reported only as a generic crash.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Cancelled(VideoQAError):
    def __init__(self, message: str = "job was cancelled") -> None:
        super().__init__("cancelled", message)

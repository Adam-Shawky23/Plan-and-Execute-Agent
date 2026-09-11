from .registry import ToolRegistry
from . import shell, files, web


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        name="run_shell_command",
        description="Run a shell command and return its stdout, stderr, and exit code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
        func=shell.run_shell_command,
    )
    registry.register(
        name="read_file",
        description="Read and return the contents of a text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
        func=files.read_file,
    )
    registry.register(
        name="write_file",
        description="Write text content to a file, overwriting it if it already exists.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Content to write to the file."},
            },
            "required": ["path", "content"],
        },
        func=files.write_file,
    )
    registry.register(
        name="list_dir",
        description="List the contents of a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path.", "default": "."},
            },
            "required": [],
        },
        func=files.list_dir,
    )
    registry.register(
        name="web_search",
        description="Search the web via DuckDuckGo and return matching results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Maximum results to return.", "default": 5},
            },
            "required": ["query"],
        },
        func=web.web_search,
    )
    registry.register(
        name="fetch_url",
        description="Fetch a URL and return its extracted text content.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
        func=web.fetch_url,
    )

    return registry

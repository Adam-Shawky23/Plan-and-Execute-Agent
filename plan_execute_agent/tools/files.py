from pathlib import Path


def read_file(path: str) -> str:
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def list_dir(path: str = ".") -> str:
    entries = sorted(
        p.name + ("/" if p.is_dir() else "") for p in Path(path).iterdir()
    )
    return "\n".join(entries) if entries else "(empty directory)"

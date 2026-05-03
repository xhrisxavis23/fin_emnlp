from pathlib import Path

try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    yaml = None

from core.utils import SingletonBaseClass


class Prompts(SingletonBaseClass, dict[str, str]):
    def __init__(self, file_path: Path) -> None:
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            raw = file.read()
        if yaml is not None:
            prompt_yaml_dict = yaml.safe_load(raw)
        else:
            prompt_yaml_dict = _safe_load_minimal_yaml(raw)

        if prompt_yaml_dict is None:
            error_message = f"Failed to load prompts from {file_path}"
            raise ValueError(error_message)

        for key, value in prompt_yaml_dict.items():
            self[key] = value


def _safe_load_minimal_yaml(text: str) -> dict:
    """
    Minimal YAML loader for the prompt files in this repo:
    - top-level and nested mappings via indentation
    - block scalars (|, |-, >, >-) as multi-line strings
    This is *not* a full YAML implementation.
    """

    def _strip_comment(line: str) -> str:
        if "#" not in line:
            return line
        # keep '#' inside block scalars; this parser only strips at line-level
        return line.split("#", 1)[0].rstrip()

    lines = [_strip_comment(l.rstrip("\n")) for l in text.splitlines()]
    root: dict = {}
    stack: list[tuple[int, dict]] = [(0, root)]
    i = 0

    def _current() -> dict:
        return stack[-1][1]

    while i < len(lines):
        raw_line = lines[i]
        if not raw_line.strip():
            i += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        if ":" not in line:
            i += 1
            continue

        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.lstrip()

        if rest in {"|", "|-", ">", ">-"}:
            block_indent = None
            block_lines: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    block_lines.append("")
                    i += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                if block_indent is None:
                    block_indent = nxt_indent
                if nxt_indent < (block_indent or 0):
                    break
                block_lines.append(nxt[(block_indent or 0) :])
                i += 1
            val = "\n".join(block_lines)
            _current()[key] = val
            continue

        if rest == "":
            new_dict: dict = {}
            _current()[key] = new_dict
            stack.append((indent + 1, new_dict))
            i += 1
            continue

        _current()[key] = rest
        i += 1

    return root

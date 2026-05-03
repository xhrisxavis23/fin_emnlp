# util/json_utils.py
import json
from typing import Any


def to_json_str(obj: Any, ensure_ascii: bool = False) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii)


def from_json_str(s: str) -> Any:
    return json.loads(s)

def strip_code_fence(s: str) -> str:
    """
    LLM이 ```json ... ``` 형태로 감싼 응답을 줄 때,
    코드 펜스를 제거해서 순수 JSON만 남긴다.
    """
    if s is None:
        return ""
    text = s.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    # 첫 줄: ``` 또는 ```json 인지 확인
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    # 마지막 줄: ``` 인지 확인
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
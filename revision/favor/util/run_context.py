# util/run_context.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import polars as pl


class NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class RunContext:
    run_id: str
    root_dir: Path
    _log_file: Path | None = None

    @classmethod
    def create(cls, base_dir: Path | str = "runs") -> "RunContext":
        base_dir = Path(base_dir)
        # 1) honor explicit FAVOR_RUN_ID (set by sweep runners to label runs deterministically)
        # 2) otherwise use microsecond precision so parallel forks never share a run_id
        import os
        env_run_id = os.environ.get("FAVOR_RUN_ID")
        if env_run_id:
            run_id = env_run_id
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = base_dir / run_id

        # 하위 디렉토리들 미리 생성
        for sub in [
            Path("logs"),
            Path("logs/agents"),
            Path("specs"),
            Path("data"),
            Path("agents"),  # 과거 경로 호환용
        ]:
            (root / sub).mkdir(parents=True, exist_ok=True)

        # 로그 파일 경로 설정
        log_file = root / "logs" / "run.log"

        return cls(run_id=run_id, root_dir=root, _log_file=log_file)

    def _resolve_path(self, rel_path: str | Path) -> Path:
        """
        Logs/agent 기록을 한 군데로 모으기 위해 'agents/' 경로를 'logs/agents/'로 리매핑한다.
        그 외 경로는 그대로 사용한다.
        """
        rel_path = Path(rel_path)
        if rel_path.parts and rel_path.parts[0] == "agents":
            rel_path = Path("logs") / rel_path
        return self.root_dir / rel_path

    # ---------- JSON 저장 ----------

    def save_json(self, rel_path: str, obj: Any) -> None:
        """
        obj: dict, list, pydantic model(.model_dump()) 등
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # pydantic BaseModel 자동 처리
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")  # pydantic v2-safe

        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def save_json_with_iter(self, rel_path: str, outer_iter: int, obj: Any) -> None:
        """
        Outer iteration별로 JSON 파일에 저장.
        기존 파일이 있으면 읽어서 outer_iter_N 키로 데이터 추가.

        Args:
            rel_path: 파일 경로
            outer_iter: Outer iteration 번호 (1부터 시작)
            obj: 저장할 객체
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # pydantic BaseModel 자동 처리
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")

        # 기존 파일 읽기
        existing_data = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # 파일이 손상되었으면 새로 시작
                existing_data = {}

        # outer_iter_N 키로 데이터 추가
        iter_key = f"outer_iter_{outer_iter}"
        existing_data[iter_key] = obj

        # 저장
        with path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def save_jsonl(self, rel_path: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """
        여러 레코드를 jsonl 형식으로 저장하고 싶을 때.
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                if hasattr(row, "model_dump"):
                    row = row.model_dump(mode="json")
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    # ---------- Polars 저장 ----------

    def save_parquet(self, rel_path: str, df: pl.DataFrame) -> None:
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    def save_parquet_with_iter(self, rel_path: str, outer_iter: int, df: pl.DataFrame) -> None:
        """
        Outer iteration별로 parquet 파일 저장.
        파일명에 _iter_N을 추가하여 저장.

        Args:
            rel_path: 파일 경로 (예: "data/price_with_formulas.parquet")
            outer_iter: Outer iteration 번호 (1부터 시작)
            df: 저장할 DataFrame
        """
        rel_path_obj = Path(rel_path)
        stem = rel_path_obj.stem
        suffix = rel_path_obj.suffix
        parent = rel_path_obj.parent

        # 파일명에 _iter_N 추가
        new_filename = f"{stem}_iter_{outer_iter}{suffix}"
        new_rel_path = parent / new_filename

        path = self._resolve_path(new_rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    def save_csv_head(self, rel_path: str, df: pl.DataFrame, n: int = 1000) -> None:
        """
        빠르게 눈으로 보고 싶은 샘플 CSV도 같이 떨어뜨리고 싶을 때.
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.head(n).write_csv(path)

    def save_csv(self, rel_path: str, df) -> None:
        """
        pandas DataFrame을 CSV로 저장.
        """
        import pandas as pd
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(df, pd.DataFrame):
            df.to_csv(path, index=False)
        elif isinstance(df, pl.DataFrame):
            df.write_csv(path)
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)}")

    def save_csv_with_iter(self, rel_path: str, outer_iter: int, df) -> None:
        """
        Outer iteration별로 CSV 파일 저장.
        """
        rel_path_obj = Path(rel_path)
        stem = rel_path_obj.stem
        suffix = rel_path_obj.suffix
        parent = rel_path_obj.parent
        new_filename = f"{stem}_iter_{outer_iter}{suffix}"
        new_rel_path = parent / new_filename
        self.save_csv(str(new_rel_path), df)

    # ---------- Pickle 저장 ----------

    def save_pickle(self, rel_path: str, obj: Any) -> None:
        """
        객체를 pickle로 저장 (Qlib 스타일 artifacts).
        """
        import pickle
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def save_pickle_with_iter(self, rel_path: str, outer_iter: int, obj: Any) -> None:
        """
        Outer iteration별로 pickle 파일 저장.
        """
        rel_path_obj = Path(rel_path)
        stem = rel_path_obj.stem
        suffix = rel_path_obj.suffix
        parent = rel_path_obj.parent
        new_filename = f"{stem}_iter_{outer_iter}{suffix}"
        new_rel_path = parent / new_filename
        self.save_pickle(str(new_rel_path), obj)

    # ---------- 텍스트 저장 ----------

    def save_text(self, rel_path: str, content: str) -> None:
        """
        텍스트 파일 저장 (마크다운, 로그 등)
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)

    def save_text_with_iter(self, rel_path: str, outer_iter: int, content: str) -> None:
        """
        Outer iteration별로 텍스트 파일에 저장.
        기존 파일이 있으면 읽어서 iteration 구분선과 함께 추가.

        Args:
            rel_path: 파일 경로
            outer_iter: Outer iteration 번호 (1부터 시작)
            content: 저장할 텍스트
        """
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 기존 내용 읽기
        existing_content = ""
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    existing_content = f.read()
            except Exception:
                existing_content = ""

        # 구분선과 함께 추가
        separator = f"\n\n{'='*80}\n"
        separator += f"OUTER ITERATION {outer_iter}\n"
        separator += f"{'='*80}\n\n"

        # 저장
        with path.open("w", encoding="utf-8") as f:
            if existing_content:
                f.write(existing_content)
                f.write(separator)
            else:
                # 첫 번째 iteration이면 구분선 추가
                f.write(separator)
            f.write(content)

    # ---------- LLM Usage 저장 ----------

    def save_llm_usage(self) -> None:
        """
        LLM 사용량 통계를 저장한다.
        전역 LLMUsageTracker에서 현재 누적된 통계를 가져와 저장.
        """
        from util.llm_tracker import get_tracker

        tracker = get_tracker()
        summary = tracker.get_summary()

        # 요약 저장
        self.save_json("specs/llm_usage.json", summary)

        # 상세 기록 저장 (최근 1000개)
        detailed = tracker.get_detailed_records(limit=1000)
        self.save_json("specs/llm_usage_detailed.json", {
            "total_records": len(tracker.stats.calls),
            "showing_last": len(detailed),
            "records": detailed,
        })

    # ---------- 로그 기록 ----------

    def log(self, message: str, print_to_console: bool = True) -> None:
        """
        로그 메시지를 콘솔에 출력하고 파일에 저장한다.

        Args:
            message: 로그 메시지
            print_to_console: 콘솔에도 출력할지 여부 (기본값: True)
        """
        # 콘솔 출력
        if print_to_console:
            print(message)

        # 파일 저장
        if self._log_file is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")

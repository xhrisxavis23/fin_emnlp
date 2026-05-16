"""
Pipeline Configuration

Central configuration definitions for the pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
# util/config.py
from pydantic import BaseModel
from pathlib import Path
import polars as pl
import pandas as pd

# ┌──────────────────────────────────────────────────────────────────────────┐
# │                           PipelineConfig Guide                            │
# ├──────────────────────────────────────────────────────────────────────────┤
# └──────────────────────────────────────────────────────────────────────────┘


class LLMConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.7  # Increased for diversity (was 0.3)
    max_tokens: int = 2048


class QlibConfig(BaseModel):
    """Qlib data source and backtest settings."""
    # Data source
    use_qlib_data: bool = True
    qlib_market: str = "csi500"                 # Qlib market: csi500, csi300, all (cn) / sp500 (us)
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"                          # "cn" or "us"
    csv_dir: str = ""                           # For US market: "~/.qlib/my_sp500_csv"

    # Transaction costs (Exchange settings)
    # CN (CSI500): open_cost=0.0005, close_cost=0.0015
    # US (SP500): open_cost=0, close_cost=0.0005, min_cost=0
    open_cost: float = 0.0005                   # 0.05% buy fee (CSI500 default)
    close_cost: float = 0.0015                  # 0.15% sell fee (CSI500 default)
    min_cost: float = 5.0                       # Minimum transaction cost (CNY/USD)
    init_cash: float = 1e8                      # Initial cash (CNY/USD)

    # Exchange settings
    limit_threshold: Optional[float] = 0.095   # Daily price limit threshold (CN: 0.095, US: None)
    # NOTE: Currently not wired into Stage4 implementations (they set deal_price explicitly).
    deal_price: str = "open"                   # Deal price hint (e.g., "open" for T+1 open execution)
    benchmark: str = "SH000905"                # Benchmark code: SH000905 (CSI500) / ^GSPC (S&P500)


class DataConfig(BaseModel):
    """Data column mapping for the pipeline."""
    # Unused fields are kept as commented placeholders for future extension.
    # US_market: str = "sp500"  # S&P500 data, not downloaded yet

    date_col: str = "timestamp"
    asset_col: str = "ticker"

    # Price columns (English column names expected)
    price_col: str = "close"                    # Close
    open_col: str = "open"                      # Open
    high_col: str = "high"                      # High
    low_col: str = "low"                        # Low
    volume_col: str = "volume"                  # Volume
    trading_value_col: str = "tradingvalue"     # Turnover value
    market_cap_col: str = "marketcap"           # Market cap (placeholder by default)
    shares_outstanding_col: str = "sharesoutstanding"  # Shares outstanding (placeholder by default)

    # fundamental_path: str | None = None


class Stage1Config(BaseModel):
    allowed_ohlcv_columns: List[str] = ["open", "high", "low", "close", "volume"]
    refine_rounds: int = 10


class Stage2Config(BaseModel):
    n_quantiles: int = 5
    monotonicity_threshold: float = 0.8


class Stage3Config(BaseModel):
    # Forward return horizon used when the hypothesis does not specify a horizon.
    horizon_days: int = 5
    # Monotonicity threshold for strictness-level improvement score.
    monotonicity_threshold: float = 0.7
    # Default strictness grid in (0, 1): larger => stricter (fewer events).
    # (higher_is_more_true uses >= quantile(t); lower_is_more_true uses <= quantile(1-t))
    strictness_grid: dict[str, float] = {
        "very_loose": 0.1,
        "loose": 0.3,
        "medium": 0.5,
        "strict": 0.7,
        "very_strict": 0.9,
    }
    # Progressive random grid settings
    # If True, use progressive random grid instead of fixed strictness_grid
    use_random_grid: bool = True
    # Number of strictness levels to generate when using random grid
    random_grid_steps: int = 3

    # Combination filtering thresholds (2-tier system)
    # A. Primary filter: ticker-level pass rate
    combination_pass_rate_threshold: float = 0.5  # 50% of tickers must pass
    # B. Secondary filter (deprecated): used to require a minimum ΔS2 (first_s2_ratio - last_s2_ratio).
    # Stage3 now uses monotonic decrease of cross-ticker aggregated S2_ratio (no magnitude threshold),
    # but we keep this field for backward-compatible config loading.
    combination_s2_improvement_threshold: float = 0.01

    # Parallelism control (used by stage3_new.py)
    # - None or <=0: auto (CPU cores - 1)
    # - 1: sequential (debug-friendly)
    # - N: use N worker processes
    n_processes: int | None = 8


class Stage4Config(BaseModel):
    """Stage 4 backtest settings."""
    # Enable/disable Optuna optimization
    enable_optuna: bool = True   # If False, only fixed_quantiles are evaluated
    n_trials: int = 20
    threshold_min: float = 0.55
    threshold_max: float = 0.95
    # Optional fixed strictness evaluation (quantile thresholds in (0,1)).
    # Example: [0.8, 0.85, 0.9, 0.95] -> fixed_q80, fixed_q85, fixed_q90, fixed_q95.
    fixed_quantiles: List[float] = [0.9]  # Single quantile for faster backtest

    # Combination evaluation
    # - <=0: evaluate all combinations
    # - N (>0): evaluate top-N combinations after sorting (see criterion below)
    max_combinations_to_evaluate: int = -1
    # Sorting criterion for selecting top-N combinations (only used when max_combinations_to_evaluate > 0).
    # - "s2_improvement": improvement in S2 ratio (default; structural validity)
    # - "mean_return": in-sample mean return (simple return)
    # - "information_ratio"/"ir"/"sharpe": deprecated; Stage3 does not provide IR/Sharpe directly
    # - "pass_rate": Stage3 pass rate (reliability)
    combination_selection_criterion: str = "s2_improvement"

    # Parallelism control
    # - None or <=0: auto (CPU cores - 1)
    # - 1: sequential (debug-friendly)
    # - N: use N worker processes
    n_processes: int | None = 8

    # Optuna logging (useful when Stage4 runs long).
    # - 0: disable per-trial logs (default; avoids noisy multi-process output)
    # - N: print a short status line every N trials (per combination)
    optuna_log_every: int = 0

    # Combined strategy (stage4_new_all.py)
    # When building a single final strategy by OR-ing multiple combos, use this global quantile for
    # all formulas' per-ticker thresholds (computed on IS, applied to OOS).
    combined_signal_q: float = 0.9

    # Position management
    horizon_days: int = 5
    lookback_window: int = 20
    # Stop loss (entry-to-now return / holding-period return): (close / entry_price) - 1.0. Set to 0.0 for break-even stop, or None to disable.
    stop_loss_threshold: float = -0.05

    # Optional entry confirmation filter to avoid "catching any down move".
    # This is applied AFTER formula-based signal generation.
    #
    # - entry_confirm_rule:
    #   - "none": no confirmation (default; backward-compatible)
    #   - "close_pos": require close to be in upper part of the day's range
    #   - "up_day": require positive 1-day return
    #   - "up_day_and_close_pos": require both
    # - entry_confirm_lag_days:
    #   - 0: apply confirmation on the same day as the formula signal
    #   - 1: require yesterday had the formula signal AND today passes confirmation (common for rebound setups)
    entry_confirm_rule: str = "none"
    entry_confirm_lag_days: int = 0
    entry_close_pos_min: float = 0.7
    entry_daily_return_min: float = 0.0

    # Trigger-exit settings
    ref_price_fn: str = "max_high"              # max_high, min_low, q50_close
    trigger_price_field: str = "high"           # high, low, close
    trigger_op: str = "gte"                     # gte, lte
    trigger_kmin: int = 1                       # First day offset to start trigger checks
    trigger_kmax: Optional[int] = None          # Last day offset to check (None => horizon_days)

    # Native Qlib strategy
    # - "trigger_exit": TriggerExitStrategy (our custom strategy: horizon_days + stop_loss)
    # - "topk_dropout": Qlib TopkDropoutStrategy (keep top-k instruments, drop n each rebalance)
    native_strategy: str = "trigger_exit"
    # TopkDropoutStrategy parameters
    topk: int = 50                              # Portfolio size (number of instruments)
    n_drop: int = 5                             # Number of instruments to drop per rebalance
    hold_thresh: int = 1                        # Minimum holding threshold


class DataSplitConfig(BaseModel):
    """
    Train / Validation / Test 데이터 분리 설정.

    - Stage 1: 전체 데이터로 수식값 계산 (수식 정의는 LLM이 생성하므로 data leakage 없음)
    - Stage 2, 3: Train 데이터로 수식 검증/선택
    - Stage 4: Validation으로 Optuna 최적화, Test로 최종 평가
    """
    # Train 기간 (수식 검증용)
    train_start: str = "2015-01-01"
    train_end: str = "2019-12-31"
    # Validation 기간 (Optuna 최적화용)
    val_start: str = "2020-01-01"
    val_end: str = "2020-12-31"
    # Test 기간 (최종 OOS 평가용)
    test_start: str = "2021-01-01"
    test_end: str = "2025-12-31"

    # 하위 호환성 속성 (기존 코드와의 호환을 위해)
    @property
    def in_sample_start(self) -> str:
        """Stage 2/3용 - train 시작"""
        return self.train_start

    @property
    def in_sample_end(self) -> str:
        """Stage 2/3용 - train 종료 (Optuna는 val 사용)"""
        return self.train_end

    @property
    def out_sample_start(self) -> str:
        """최종 평가용 - test 시작"""
        return self.test_start

    @property
    def out_sample_end(self) -> str:
        """최종 평가용 - test 종료"""
        return self.test_end


class PipelineControlConfig(BaseModel):
    """
    Stage execution toggles.

    - enable_stage2: if False, skip Stage 2 (formula validation) and treat all formulas as PASS
    - enable_stage3: if False, skip Stage 3 (combination validation) and treat all combinations as PASS
    """
    enable_stage2: bool = True   # formula_validation
    enable_stage3: bool = True   # combination_validation


class RefinementConfig(BaseModel):
    """
    Refinement-loop configuration (iterative hypothesis improvement).

    Two loops:
    1) Inner loop: Stage1 ⇄ Stage2
       - Formula-level refinement
       - Refine formulas that failed Stage 2 and re-validate

    2) Outer loop: Stage4 → Stage1
       - Hypothesis-level refinement
       - Regenerate the hypothesis based on Stage 4 outcomes
       - Repeat up to max_outer_iterations
    """
    # Inner loop (Stage1 ⇄ Stage2)
    enable_inner_loop: bool = True
    max_inner_iterations: int = 3  # Max retries when Stage2 has failures

    # Outer loop (Stage4 → Stage1)
    enable_outer_loop: bool = True  # Enable hypothesis refinement loop
    max_outer_iterations: int = 5  # Max full-pipeline iterations


class RDConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    data: DataConfig = DataConfig()
    qlib: QlibConfig = QlibConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()
    stage4: Stage4Config = Stage4Config()
    data_split: DataSplitConfig = DataSplitConfig()
    refinement: RefinementConfig = RefinementConfig()
    pipeline_control: PipelineControlConfig = PipelineControlConfig()


def _load_env_from_dotenv(dotenv_path: Path) -> None:
    """
    Minimal .env loader (no external dependency).

    - Only sets keys that are not already present in the environment.
    - Supports `KEY=VALUE` lines (optionally prefixed with `export `).
    - Ignores blank lines and comments (# ...).
    """
    try:
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            val = val.strip().strip("\"'").strip()
            if not val:
                continue
            os.environ.setdefault(key, val)
    except Exception:
        # .env loading is best-effort; failures must not block the pipeline.
        return


def load_rd_config() -> RDConfig:
    # Load OpenAI API key etc. from repo-root .env if present.
    _load_env_from_dotenv(Path(__file__).resolve().parents[1] / ".env")

    # Load market-specific configuration from environment variable
    import os
    market = os.getenv("MARKET", "cn").lower()  # Default: CN market

    config = RDConfig()

    if market == "cn":
        # CN Market (CSI500) configuration
        config.qlib.qlib_market = "csi500"
        config.qlib.region = "cn"
        config.qlib.provider_uri = os.getenv("FAVOR_QLIB_PROVIDER_URI_CN", "~/.qlib/qlib_data/cn_data")
        config.qlib.open_cost = 0.0005      # 0.05% buy fee
        config.qlib.close_cost = 0.0015     # 0.15% sell fee
        config.qlib.min_cost = 5.0          # Minimum transaction cost (CN default)
        config.qlib.limit_threshold = 0.095  # Daily price limit threshold (Qlib default)
        config.qlib.benchmark = "SH000905"  # CSI500 index
    else:
        # US Market (S&P500) configuration
        config.qlib.qlib_market = "sp500"
        config.qlib.region = "us"
        config.qlib.provider_uri = os.getenv("FAVOR_QLIB_PROVIDER_URI_US", "~/.qlib/sh_sp500_qlib")
        config.qlib.open_cost = 0            # No buy fee
        config.qlib.close_cost = 0.0005      # 0.05% sell fee
        config.qlib.min_cost = 0.0           # No minimum cost
        config.qlib.limit_threshold = None   # No daily price limit
        config.qlib.benchmark = "^GSPC"      # S&P500 index

    # ─── Additive env-var overrides for sweep / repro experiments ───────────
    # All optional; when unset, fall back to the existing defaults above.
    def _env_str(name: str, current: str) -> str:
        v = os.getenv(name)
        return v if v is not None and v != "" else current

    def _env_float(name: str, current: float) -> float:
        v = os.getenv(name)
        if v is None or v == "":
            return current
        try:
            return float(v)
        except ValueError:
            return current

    def _env_int(name: str, current: int) -> int:
        v = os.getenv(name)
        if v is None or v == "":
            return current
        try:
            return int(v)
        except ValueError:
            return current

    def _env_optional_float(name: str, current):
        """Same as _env_float but allows the literal string 'none' / 'disable' / 'null' to set None."""
        v = os.getenv(name)
        if v is None or v == "":
            return current
        if v.strip().lower() in ("none", "null", "disable", "disabled", "off"):
            return None
        try:
            return float(v)
        except ValueError:
            return current

    # LLM
    config.llm.model_name = _env_str("FAVOR_LLM_MODEL", config.llm.model_name)
    config.llm.temperature = _env_float("FAVOR_LLM_TEMPERATURE", config.llm.temperature)

    # Stage 3
    config.stage3.combination_pass_rate_threshold = _env_float(
        "FAVOR_COMBO_PASS_RATE", config.stage3.combination_pass_rate_threshold
    )

    # Stage 4
    config.stage4.horizon_days = _env_int("FAVOR_HORIZON_DAYS", config.stage4.horizon_days)
    config.stage4.stop_loss_threshold = _env_optional_float(
        "FAVOR_STOP_LOSS_THRESHOLD", config.stage4.stop_loss_threshold
    )
    config.stage4.entry_confirm_rule = _env_str(
        "FAVOR_ENTRY_CONFIRM_RULE", config.stage4.entry_confirm_rule
    )
    config.stage4.native_strategy = _env_str(
        "FAVOR_NATIVE_STRATEGY", config.stage4.native_strategy
    )
    config.stage4.threshold_min = _env_float("FAVOR_THRESHOLD_MIN", config.stage4.threshold_min)
    config.stage4.threshold_max = _env_float("FAVOR_THRESHOLD_MAX", config.stage4.threshold_max)

    # data_split (sweep over time ranges; env vars unset → defaults intact)
    config.data_split.train_start = _env_str("FAVOR_TRAIN_START", config.data_split.train_start)
    config.data_split.train_end   = _env_str("FAVOR_TRAIN_END",   config.data_split.train_end)
    config.data_split.val_start   = _env_str("FAVOR_VAL_START",   config.data_split.val_start)
    config.data_split.val_end     = _env_str("FAVOR_VAL_END",     config.data_split.val_end)
    config.data_split.test_start  = _env_str("FAVOR_TEST_START",  config.data_split.test_start)
    config.data_split.test_end    = _env_str("FAVOR_TEST_END",    config.data_split.test_end)

    return config

def load_price_data(
    cfg: RDConfig,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load price data from Qlib using RDConfig settings

    Args:
        cfg: RDConfig instance
        start_time: Override start time (optional, defaults to data_split.train_start)
        end_time: Override end time (optional, defaults to data_split.test_end)

    Returns:
        Polars DataFrame with price data
    """
    if not cfg.qlib.use_qlib_data:
        raise NotImplementedError("Parquet loading is deprecated. Use Qlib data (set qlib.use_qlib_data=True)")

    print("📊 Loading data from Qlib...")

    # Get Qlib settings from cfg
    provider_uri = str(Path(cfg.qlib.provider_uri).expanduser())
    region = cfg.qlib.region
    market = cfg.qlib.qlib_market
    start_time = start_time or cfg.data_split.train_start
    end_time = end_time or cfg.data_split.test_end

    # Import Qlib from system installation (not local directory)
    import qlib
    from qlib.data import D

    # Initialize Qlib with appropriate region
    if region == "us":
        from qlib.constant import REG_US
        qlib.init(provider_uri=provider_uri, region=REG_US)
    else:
        qlib.init(provider_uri=provider_uri, region=region)
    print(f"✅ Qlib initialized: {provider_uri} (region={region}, market={market})")

    # Get instrument pool
    if region == "us" and cfg.qlib.csv_dir:
        # For US market: read symbols from CSV directory
        csv_path = Path(cfg.qlib.csv_dir).expanduser()
        symbols = [p.stem for p in csv_path.glob("*.csv")]
        instruments_pool = symbols
        print(f"📊 Loading {len(symbols)} US symbols from {cfg.qlib.csv_dir} ({start_time} to {end_time})...")
    else:
        # For CN market: use D.instruments()
        instruments_pool = D.instruments(market)
        print(f"📊 Loading {market} data from {start_time} to {end_time}...")

    # Load data using instrument pool directly
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]

    try:
        df = D.features(instruments_pool, fields=fields, start_time=start_time, end_time=end_time)
        if df is None or len(df) == 0:
            raise ValueError(f"Failed to load any data from Qlib {market}")

        # Reset index to get instrument and datetime as columns
        df = df.reset_index()

        # Rename columns
        combined_df = df.rename(columns={
            "datetime": "timestamp",
            "instrument": "ticker",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$factor": "factor",
        })

        # Count unique instruments
        n_instruments = combined_df['ticker'].nunique()
        print(f"✅ Loaded {len(combined_df):,} rows from {n_instruments} instruments")

    except Exception as e:
        raise ValueError(f"Failed to load data from Qlib {market}: {e}")

    # Convert numeric columns to float to avoid isnan type errors
    numeric_cols = ["open", "high", "low", "close", "volume", "factor"]
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # Convert to Polars
    polars_df = pl.from_pandas(combined_df)

    # Convert timestamp to YYYYMMDD string format
    polars_df = polars_df.with_columns([
        pl.col("timestamp").dt.strftime("%Y%m%d").alias("timestamp")
    ])

    # Add calculated columns
    polars_df = polars_df.with_columns([
        (pl.col("close") * pl.col("volume")).alias("tradingvalue")
    ])

    # Add placeholder columns for compatibility
    for col in ["marketcap", "sharesoutstanding"]:
        polars_df = polars_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(col)
        ])

    print(f"📊 Final DataFrame shape: {polars_df.shape}")
    print(f"   Date range: {polars_df['timestamp'].min()} ~ {polars_df['timestamp'].max()}")

    return polars_df

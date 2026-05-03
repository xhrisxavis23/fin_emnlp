import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        env_to_use: dict[str, Any] = dict(run_env or {})
        for key in ("provider_uri", "region", "market", "benchmark", "benchmark_list", "benchmark_mode"):
            if key not in env_to_use and key in os.environ:
                env_to_use[key] = os.environ[key]

        benchmark_mode = str(env_to_use.get("benchmark_mode", "")).strip()
        if benchmark_mode in {
            "equal_weight_market",
            "equal_weight_all",
            "buyhold_equal_weight_market",
            "buyhold_equal_weight_all",
        } and not env_to_use.get("benchmark_list"):
            provider_uri = env_to_use.get("provider_uri")
            market = env_to_use.get("market")
            if provider_uri and (benchmark_mode != "equal_weight_market" or market):
                provider_path = Path(str(provider_uri)).expanduser()
                instruments_file = (
                    provider_path
                    / "instruments"
                    / (
                        "all.txt"
                        if benchmark_mode in {"equal_weight_all", "buyhold_equal_weight_all"}
                        else f"{market}.txt"
                    )
                )
                if instruments_file.exists():
                    tickers: list[str] = []
                    for line in instruments_file.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tickers.append(line.split()[0])
                    if tickers:
                        env_to_use["benchmark_list"] = ",".join(tickers)

        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        # Run the Qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=env_to_use,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=env_to_use,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log

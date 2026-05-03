
import os
import numpy as np
import pandas as pd

from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol
from alphaagent.components.coder.factor_coder.function_lib import *


def calculate_factor(expr: str, name: str) -> None:
    df = pd.read_hdf('./daily_pv.h5', key='data')

    expr2 = parse_symbol(expr, df.columns)
    expr2 = parse_expression(expr2)

    for col in df.columns:
        expr2 = expr2.replace(col[1:], f"df['{col}']")

    df[name] = eval(expr2)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')


if __name__ == '__main__':
    expr = "TS_RANK(close - low, 5)"
    name = "formula009"
    calculate_factor(expr, name)

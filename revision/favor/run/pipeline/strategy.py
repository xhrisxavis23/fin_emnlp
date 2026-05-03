"""
================================================================================
Custom Strategy for Horizon Days Logic

Implemented by inheriting from Qlib `BaseStrategy`:
- Enter based on a boolean signal
- Time stop (exit after `horizon_days`)
- Stop loss (cut loss based on cumulative return vs entry price)

Prices are retrieved via Qlib's `trade_exchange`.
================================================================================
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position


@dataclass
class PositionState:
    """Per-instrument position state."""
    stock_id: str
    entry_date: pd.Timestamp
    entry_price: float
    holding_days: int = 0


@dataclass
class TriggerExitConfig:
    """Strategy configuration."""
    horizon_days: int = 5  # max holding period
    stop_loss_threshold: float = -0.05  # stop-loss threshold (cumulative return vs entry)


class TriggerExitStrategy(BaseStrategy):
    """
    Time-stop strategy (Qlib backtest integration).

    Implements the current `stage4.py::_simulate_positions()` logic as a Qlib Strategy:
    1. Enter when boolean signal is True
    2. Time stop: exit when holding_days exceeds `horizon_days`
    3. Stop loss: cut loss based on cumulative return vs entry price

    Prices are retrieved via Qlib's `trade_exchange`.
    """

    def __init__(
        self,
        *,
        signal: pd.Series,  # MultiIndex (datetime, instrument) -> bool
        config: TriggerExitConfig,
        risk_degree: float = 0.95,
        **kwargs,
    ):
        """
        Parameters
        ----------
        signal : pd.Series
            Boolean signal Series with MultiIndex (datetime, instrument)
            signal=True means entry signal
        config : TriggerExitConfig
            Strategy configuration
        risk_degree : float
            Capital allocation fraction (default: 0.95)
        """
        super().__init__(**kwargs)

        self.config = config
        self.risk_degree = risk_degree

        # Signal Series (MultiIndex: datetime, instrument)
        self.signal = signal

        # Current position states (stock_id -> PositionState)
        self._position_states: Dict[str, PositionState] = {}

        # Trade records
        self.trade_records: List[Dict[str, Any]] = []

    def _check_time_stop(self, state: PositionState) -> bool:
        """Check time-stop condition."""
        return state.holding_days >= self.config.horizon_days

    def _check_stop_loss(self, state: PositionState, current_close: float) -> bool:
        """
        Check stop-loss condition (cumulative return vs entry price).

        Parameters
        ----------
        state : PositionState
            Position state
        current_close : float
            Current close price

        Returns
        -------
        bool
            Whether stop-loss is triggered
        """
        if np.isnan(current_close) or state.entry_price <= 0:
            return False

        # Cumulative return vs entry
        cumulative_return = (current_close / state.entry_price) - 1.0

        # Stop-loss triggers when cumulative return is below threshold
        return cumulative_return <= self.config.stop_loss_threshold

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """
        Called every trading day to generate buy/sell orders.

        Logic:
        1. Check existing positions → sell if exit conditions are met
        2. Check new signals → buy if entry conditions are met
        """
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_date = pd.Timestamp(trade_start_time.date())

        sell_orders: List[Order] = []
        buy_orders: List[Order] = []

        # Current position snapshot
        current_position: Position = copy.deepcopy(self.trade_position)
        current_stock_list = current_position.get_stock_list()
        cash = current_position.get_cash()

        # ═══════════════════════════════════════════════════════════════════
        # 1) Existing positions: check exit conditions
        # ═══════════════════════════════════════════════════════════════════
        # Store as (stock_id -> exit_reason)
        stocks_to_sell: Dict[str, str] = {}

        for stock_id in list(self._position_states.keys()):
            state = self._position_states[stock_id]
            state.holding_days += 1  # increment holding days

            # Fetch current price
            try:
                current_close = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
            except Exception:
                current_close = np.nan

            exit_reason = None

            # Stop-loss check (highest priority)
            if self._check_stop_loss(state, current_close):
                exit_reason = "stop_loss"
            # Time-stop check
            elif self._check_time_stop(state):
                exit_reason = "time_stop"

            if exit_reason:
                stocks_to_sell[stock_id] = exit_reason

        # Build sell orders
        stocks_sold: List[str] = []  # instruments for which a sell order is actually created

        for stock_id in stocks_to_sell:
            if stock_id not in current_stock_list:
                continue

            # Check tradability
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            ):
                continue

            sell_amount = current_position.get_stock_amount(code=stock_id)
            if sell_amount <= 0:
                continue

            sell_order = Order(
                stock_id=stock_id,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            )

            if not self.trade_exchange.check_order(sell_order):
                continue

            sell_orders.append(sell_order)
            stocks_sold.append(stock_id)

            # Estimate proceeds
            sell_price = self.trade_exchange.get_deal_price(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            )
            if sell_price is not None and not np.isnan(sell_price):
                estimated_value = sell_amount * sell_price
                cash += estimated_value

        # Remove sold instruments from state + record trades
        for stock_id in stocks_sold:
            state = self._position_states.get(stock_id)
            if state is None:
                continue

            exit_reason = stocks_to_sell[stock_id]

            # Fetch sell price
            try:
                exit_price = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
            except Exception:
                exit_price = np.nan

            if np.isnan(exit_price):
                exit_price = state.entry_price

            # Trade record (only when a sell order is actually created)
            self.trade_records.append({
                'ticker': stock_id,
                'entry_date': state.entry_date.strftime('%Y-%m-%d'),
                'entry_price': state.entry_price,
                'exit_date': current_date.strftime('%Y-%m-%d'),
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'holding_days': state.holding_days,
                'return_pct': (exit_price / state.entry_price - 1) if state.entry_price > 0 else 0.0,
            })

            del self._position_states[stock_id]

        # ═══════════════════════════════════════════════════════════════════
        # 2) New entries: instruments with signal=True
        # ═══════════════════════════════════════════════════════════════════
        # From today's signal=True instruments, keep those without an existing position
        try:
            today_signals = self.signal.xs(current_date, level=0)
            new_entries = today_signals[today_signals == True].index.tolist()
        except KeyError:
            new_entries = []

        # Exclude instruments already held or sold today
        new_entries = [
            s for s in new_entries
            if s not in self._position_states and s not in stocks_to_sell
        ]

        if new_entries:
            # Equal-weight allocation
            available_cash = cash * self.risk_degree
            weight_per_stock = available_cash / len(new_entries) if new_entries else 0

            for stock_id in new_entries:
                # Check tradability
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                ):
                    continue

                # Fetch buy price
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )

                if buy_price is None or buy_price <= 0 or np.isnan(buy_price):
                    continue

                buy_amount = weight_per_stock / buy_price

                # Round to trade unit
                factor = self.trade_exchange.get_factor(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)

                if buy_amount <= 0:
                    continue

                buy_order = Order(
                    stock_id=stock_id,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )

                # Add order after validation
                if not self.trade_exchange.check_order(buy_order):
                    continue

                buy_orders.append(buy_order)

                # Add position state (only when order is valid)
                self._position_states[stock_id] = PositionState(
                    stock_id=stock_id,
                    entry_date=current_date,
                    entry_price=buy_price,
                    holding_days=0,
                )

        return TradeDecisionWO(sell_orders + buy_orders, self)

    def reset(self, **kwargs):
        """Reset strategy state."""
        super().reset(**kwargs)
        self._position_states = {}
        self.trade_records = []

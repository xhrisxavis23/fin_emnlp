#!/bin/bash
################################################################################
# CN Market (CSI500) Pipeline Runner
#
# Usage:
#   ./run_cn.sh                                      # Default concept
#   ./run_cn.sh "Short-term mean reversion after panic selling"              # Custom concept
#   ./run_cn.sh "Short-term mean reversion after panic selling" --outer-loop 5     # With outer loop
#   ./run_cn.sh "Short-term Downside Mean-reversion" --outer-loop 1     # With outer loop
#   ./run_cn.sh "Short-term Sell-off Rebound" --outer-loop 1            # Example that tends to produce stable results
#   ./run_cn.sh "Short-term Upside Momentum" --outer-loop 1             # With outer loop

#   ./run_cn.sh "In a strong uptrend, when price pulls back to the 20-day moving average, buying near that level increases the probability of price retesting the previous high." --outer-loop 5             # With outer loop
#   ./run_cn.sh "After a breakout to a new high, a pullback toward the 20-day moving average often serves as support, increasing the probability of price revisiting the breakout level or exceeding it." --outer-loop 5             # With outer loop
#   ./run_cn.sh "In a strong uptrend, pullbacks toward the 20-day moving average accompanied by compressed volatility increase the probability of an upside continuation." --outer-loop 5             # With outer loop
#   ./run_cn.sh "In a strong uptrend, when price pulls back to the 20-day moving average, the probability of price rebounding to the previous high within the next 5–10 trading days increases." --outer-loop 5             # With outer loop
#   ./run_cn.sh "After a sharp sell-off, stocks that close near the day’s high, indicating strong intraday recovery, are more likely to rebound over the next 3 trading days." --outer-loop 5
#   ./run_cn.sh "Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days." --outer-loop 5
#   ./run_us.sh "In a strong uptrend, when price pulls back to the 20-day moving average, buying near that level increases the probability of price retesting the previous high." --outer-loop 5             # With outer loop

################################################################################

set -e  # Exit on error

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "🇨🇳 CN Market (CSI500) Configuration"
echo "=================================================="
echo "✅ CN market configuration loaded via MARKET=cn"
echo ""

# Run pipeline with all arguments passed through
echo "🚀 Running pipeline..."
export MARKET=cn
python run_pipeline.py "$@"

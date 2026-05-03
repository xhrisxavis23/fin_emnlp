#!/bin/bash
################################################################################
# US Market (SP500) Pipeline Runner
#
# Usage:
#   ./run_us.sh                                      # Default concept
#   ./run_us.sh "Short-term mean reversion after panic selling"               # Custom concept
#   ./run_us.sh "Short-term mean reversion after panic selling" --outer-loop 5     # With outer loop
#   ./run_us.sh "Following a prior price decline, some stocks enter a compressed trading state characterized by reduced intraday range and the absence of further downside follow-through. When selling pressure becomes exhausted while latent demand continues to absorb supply, prices may remain temporarily stable despite elevated participation. This imbalance can resolve abruptly, leading to a sharp upward price movement over the subsequent few trading days." --outer-loop 5

################################################################################

set -e  # Exit on error

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "🇺🇸 US Market (SP500) Configuration"
echo "=================================================="
echo "✅ US market configuration loaded via MARKET=us"
echo ""

# Run pipeline with all arguments passed through
echo "🚀 Running pipeline..."
export MARKET=us
export PYTHONWARNINGS=ignore
python run_pipeline.py "$@"

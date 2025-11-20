#!/usr/bin/env bash

T_CYCLE_147="$1"
T_CYCLE_148="$2"

# check inputs
if [ -z "$T_CYCLE_147" ] || [ -z "$T_CYCLE_148" ]; then
    echo "Usage: $0 <t-cycle-for-A147> <t-cycle-for-A148>"
    exit 1
fi

echo "Running A147 with t_cycle = $T_CYCLE_147"
uv run optim mc-exp \
    --t-cycle "$T_CYCLE_147" \
    --t-exp 57600 \
    --rate 10.98 \
    --tp 0.894 \
    --td 4.06 \
    --eff1 1 --eff2 1 \
    --samples 10 \
    --a 147 \
echo
echo "Running A148 with t_cycle = $T_CYCLE_148"
uv run optim mc-exp \
    --t-cycle "$T_CYCLE_148" \
    --t-exp 111600 \
    --rate 0.99 \
    --tp 0.619 \
    --td 1.411 \
    --eff1 1 --eff2 1 \
    --samples 10 \
    --a 148 \

#!/usr/bin/env bash
# Min-Mem session bootstrap — loads dictionary and records session metrics.
set -euo pipefail
input=$(cat)
python3 -m min_mem.hooks.session_start <<<"$input" 2>/dev/null || echo '{}'
exit 0

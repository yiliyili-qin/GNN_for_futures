#!/usr/bin/env bash
# Local only. Example (does not run unless explicitly invoked):
# TEN_YEAR_PYTHON=/path/to/existing/python bash tools/launch_ten_year_graph_ablation.sh \
#   --background --quotes-dir /verified/quotes --output /new/local/run --execute
# Remaining arguments pass unchanged to the runner; no installs, copies or deployment.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${TEN_YEAR_PYTHON:-python}
background=false
if [[ ${1:-} == --background ]]; then
  background=true
  shift
fi
if ! "$background"; then
  exec "$PYTHON" -I -B "$ROOT/tools/run_ten_year_graph_ablation.py" "$@"
fi
output=
previous=
for argument in "$@"; do
  if [[ "$previous" == --output ]]; then output=$argument; fi
  if [[ "$argument" == --output=* ]]; then output=${argument#--output=}; fi
  previous=$argument
done
[[ -n "$output" && ! -e "$output" && ! -e "$output.launcher" ]] || {
  echo 'Background mode requires a new --output and companion .launcher directory' >&2
  exit 2
}
mkdir -p "$(dirname "$output")"
mkdir "$output.launcher"
nohup "$PYTHON" -I -B "$ROOT/tools/run_ten_year_graph_ablation.py" "$@" > "$output.launcher/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.launcher/pid"
printf 'STARTED local_pid=%s log=%s status=%s\n' "$pid" "$output.launcher/run.log" "$output/run_status.json"

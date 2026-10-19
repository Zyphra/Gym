#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

# Pin must match mini_swe_agent_local/app.py's DefaultAgent and model API.
MINI_REQ="$NEMO_GYM_ROOT/responses_api_agents/mini_swe_agent_local/requirements.txt"
MINI_SPEC="${MINI_SPEC:-$(sed -n 's/^\(mini-swe-agent==.*\)$/\1/p' "$MINI_REQ")}"
: "${MINI_SPEC:?could not read the mini-swe-agent pin from $MINI_REQ}"

install_portable_python
install_nemo_gym_deps

echo "Installing $MINI_SPEC"
install_python_packages "$MINI_SPEC"

"$DEPS_DIR/bin/python3" -c "from minisweagent.agents.default import DefaultAgent; from minisweagent.environments.local import LocalEnvironment; print('mini-swe-agent OK')"

echo "mini_swe_agent_local deps ready at $DEPS_DIR"

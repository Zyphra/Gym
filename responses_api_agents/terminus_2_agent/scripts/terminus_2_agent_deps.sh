#!/bin/bash
# Install Terminus-2 deps into $DEPS_DIR for anyterminal_agent.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

TERMINUS_REQ="$NEMO_GYM_ROOT/responses_api_agents/terminus_2_agent/requirements.txt"
HARBOR_SPEC="${HARBOR_SPEC:-$(sed -n 's/^harbor @ //p' "$TERMINUS_REQ")}"
: "${HARBOR_SPEC:?could not read the Harbor pin from $TERMINUS_REQ}"
TIKTOKEN_SPEC="$(grep '^tiktoken' "$TERMINUS_REQ")"

install_portable_python
install_nemo_gym_deps

echo "Installing Harbor Terminus-2 ($HARBOR_SPEC)"
"$DEPS_DIR/bin/python3" -m pip install "$HARBOR_SPEC" "$TIKTOKEN_SPEC"
"$DEPS_DIR/bin/python3" -c \
    "from harbor.agents.terminus_2.terminus_2 import Terminus2; print(Terminus2.name())"
# The agent imports inside task images; litellm guards its optional OCI cryptography import.
PYTHONPATH="$NEMO_GYM_ROOT" "$DEPS_DIR/bin/python3" "$SCRIPT_DIR/check_glibc_floor.py" \
    responses_api_agents.terminus_2_agent.app --floor 2.17 --optional cryptography

echo "terminus_2_agent deps ready at $DEPS_DIR"

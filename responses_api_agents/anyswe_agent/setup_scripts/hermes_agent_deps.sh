#!/bin/bash
# Hermes' runtime is prepared by the Hermes agent's own deps script, with this harness's portable Python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
PORTABLE_PYTHON_SH="$SCRIPT_DIR/_portable_python.sh" exec bash "$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/scripts/hermes_agent_deps.sh"

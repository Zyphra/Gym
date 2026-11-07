#!/bin/bash
# Install hermes_agent deps into $DEPS_DIR (mounted read-only at /agent_deps_mount).
#
# Hermes runs unmodified: the pinned upstream source tree is placed in $DEPS_DIR/hermes-agent and put
# on the path with a relative .pth entry (Hermes refuses wheel and sdist builds; its supported install
# is a checkout plus its locked dependencies). Dependencies come from Hermes' own uv.lock, then Gym's,
# so Gym's audited OpenAI SDK pin wins where the two pin different versions.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Hermes' dependency pins apply on Python >= 3.14 only.
export PYTHON_VERSION="${PYTHON_VERSION:-3.14.7}"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"
command -v uv >/dev/null || { echo "uv is required to read Hermes' uv.lock" >&2; exit 1; }

# The pin is the commented `hermes-agent @ <repo>@<commit>` line of requirements.txt.
HERMES_REQ="$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/requirements.txt"
HERMES_SPEC="$(sed -n 's/^# hermes-agent @ //p' "$HERMES_REQ")"
HERMES_REPO="${HERMES_SPEC%@*}"
HERMES_COMMIT="${HERMES_SPEC##*@}"
: "${HERMES_COMMIT:?could not read the hermes-agent pin from $HERMES_REQ}"

install_portable_python

HERMES_DIR="$DEPS_DIR/hermes-agent"
if [ ! -f "$HERMES_DIR/run_agent.py" ]; then
    mkdir -p "$HERMES_DIR"
    curl -fsSL "${HERMES_REPO/github.com/codeload.github.com}/tar.gz/$HERMES_COMMIT" | tar xz -C "$HERMES_DIR" --strip-components=1
fi
echo "$HERMES_COMMIT" > "$HERMES_DIR/.hermes_commit"

LOCKED="$DEPS_DIR/.hermes-locked-requirements.txt"
uv export --project "$HERMES_DIR" --frozen --no-hashes --no-emit-project --no-dev --format requirements-txt > "$LOCKED"
install_python_packages --no-deps -r "$LOCKED"
install_nemo_gym_deps

SITE="$("$DEPS_DIR/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
# Relative to site-packages, so the runtime stays valid wherever $DEPS_DIR is mounted.
"$DEPS_DIR/bin/python3" -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$HERMES_DIR" "$SITE" > "$SITE/hermes-agent.pth"

# Isolated (-I): the current directory may hold an unrelated `tools` package.
"$DEPS_DIR/bin/python3" -I -c "import model_tools; from run_agent import AIAgent; print('hermes-agent OK')"
echo "hermes_agent deps ready at $DEPS_DIR"

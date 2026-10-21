#!/bin/bash
# Install codex_agent deps into $DEPS_DIR: portable CPython + NeMo-Gym deps + the pinned Codex CLI.
#
# No Node: the npm package @openai/codex is a launcher for the static musl binary published in
# its linux-x64 platform package (@openai/codex@<version>-linux-x64). This installs that
# package's vendor tree (codex, its bundled rg/bwrap/zsh and code-mode host) as npm lays it out,
# after checking the registry integrity of the tarball and the binary's sha256.
#
# Network is used here only. With codex on $DEPS_DIR/bin and codex_install: offline, the agent
# never installs or downloads at run time and refuses any other version.
set -euo pipefail
set -x

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

CODEX_VERSION=0.144.4
CODEX_TARGET=x86_64-unknown-linux-musl
CODEX_TARBALL_URL="https://registry.npmjs.org/@openai/codex/-/codex-${CODEX_VERSION}-linux-x64.tgz"
CODEX_TARBALL_INTEGRITY="sha512-2jxrmV6+/7eBNdg5uhhmOEPFu2o28eYY/ClLzWhSBHH8uo3f2KA1z9JQcVtwlbToW03nEPlEzYNYfCF1UBqsVQ=="
CODEX_BINARY_SHA256="2b3edc9cdfd1717fba3dbc92817205a8a2c7511d459e456d4817eeff6f78ed7a"
# Optional pre-fetched copy of that tarball; verified the same way as a download.
CODEX_TARBALL="${CODEX_TARBALL:-}"

source "$NEMO_GYM_ROOT/responses_api_agents/anyswe_agent/setup_scripts/_portable_python.sh"

install_portable_python
install_nemo_gym_deps

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
if [ -n "$CODEX_TARBALL" ]; then
    cp "$CODEX_TARBALL" "$work/codex.tgz"
else
    curl -fsSL "$CODEX_TARBALL_URL" -o "$work/codex.tgz"
fi
integrity="sha512-$(openssl dgst -sha512 -binary "$work/codex.tgz" | base64 -w0)"
if [ "$integrity" != "$CODEX_TARBALL_INTEGRITY" ]; then
    echo "codex tarball integrity mismatch: $integrity" >&2
    exit 1
fi
tar -xzf "$work/codex.tgz" -C "$work" "package/vendor/$CODEX_TARGET"
echo "$CODEX_BINARY_SHA256  $work/package/vendor/$CODEX_TARGET/bin/codex" | sha256sum -c -

libexec="$DEPS_DIR/libexec/codex/$CODEX_VERSION"
rm -rf "$libexec"
mkdir -p "$(dirname "$libexec")" "$DEPS_DIR/bin"
cp -a "$work/package/vendor/$CODEX_TARGET" "$libexec"
# The binary finds its bundled resources next to its resolved path.
ln -sfn "../libexec/codex/$CODEX_VERSION/bin/codex" "$DEPS_DIR/bin/codex"

mkdir -p "$work/home/codex"
version="$(env -i PATH=/usr/bin:/bin HOME="$work/home" CODEX_HOME="$work/home/codex" "$DEPS_DIR/bin/codex" --version)"
if [ "$version" != "codex-cli $CODEX_VERSION" ]; then
    echo "codex --version mismatch: $version" >&2
    exit 1
fi

echo "codex_agent deps ready at $DEPS_DIR ($version)"

#!/bin/bash
# Shared helper for a relocatable CPython under $DEPS_DIR.
set -euo pipefail

# Keep pip from satisfying deps from the host user site.
export PYTHONNOUSERSITE=1

PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"
PBS_RELEASE="${PBS_RELEASE:-20260805}"
ARCH="${ARCH:-x86_64-unknown-linux-gnu}"

install_portable_python() {
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        echo "Portable python already present at $DEPS_DIR/bin/python3"
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${ARCH}-install_only.tar.gz"
    echo "Downloading portable python: $url"
    # Tarball extracts to python/{bin,lib}.
    curl -fsSL "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    case "$ARCH" in *-linux-gnu) install_glibc_launcher || return 1 ;; esac
    if portable_python_can_run; then
        install_python_packages --upgrade pip
    fi
}

# Task images without glibc (Alpine/musl) cannot exec the glibc CPython. Bundle the
# build host's glibc loader and core libraries under $DEPS_DIR/glibc and replace the
# interpreter entry points with a launcher that uses them only when the image has no
# real glibc loader. Alpine images with gcompat ship a glibc-named loader shim over musl,
# so a musl loader (/lib/ld-musl-x86_64.so.1) also selects the bundled glibc. On musl images --argv0 keeps sys.executable on the launcher, so child
# interpreters come back through it; glibc images exec the interpreter directly. A
# build host without these libraries fails the install instead of shipping a
# glibc-only runtime.
GLIBC_LOADER="${GLIBC_LOADER:-/lib64/ld-linux-x86-64.so.2}"
GLIBC_LIBS="libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libutil.so.1 libresolv.so.2 libnss_dns.so.2 libnss_files.so.2 libcrypt.so.1 libgcc_s.so.1 libstdc++.so.6 libz.so.1"

install_glibc_launcher() {
    local real
    real="$(readlink -f "$DEPS_DIR/bin/python3")"
    local name
    name="$(basename "$real")"
    [ -e "$DEPS_DIR/bin/$name.real" ] && return 0
    mkdir -p "$DEPS_DIR/glibc"
    cp -L "$GLIBC_LOADER" "$DEPS_DIR/glibc/ld-linux-x86-64.so.2"
    local lib path
    for lib in $GLIBC_LIBS; do
        path="$(ldconfig -p | awk -v l="$lib" '$1 == l && /x86-64/ && !f {print $NF; f = 1}')"
        [ -n "$path" ] || { echo "glibc launcher: host lacks $lib" >&2; return 1; }
        cp -L "$path" "$DEPS_DIR/glibc/$lib"
    done
    mv "$real" "$DEPS_DIR/bin/$name.real"
    cat > "$DEPS_DIR/bin/$name" <<LAUNCHER
#!/bin/sh
R="\${0%/bin/*}"
if [ -e "\${NGSWE_SYSROOT:-}/lib64/ld-linux-x86-64.so.2" ] && [ ! -e "\${NGSWE_SYSROOT:-}/lib/ld-musl-x86_64.so.1" ]; then exec "\$R/bin/$name.real" "\$@"; fi
exec "\$R/glibc/ld-linux-x86-64.so.2" --library-path "\$R/glibc:\$R/lib" --argv0 "\$R/bin/$name" "\$R/bin/$name.real" "\$@"
LAUNCHER
    chmod 755 "$DEPS_DIR/bin/$name"
    ln -sf "$name" "$DEPS_DIR/bin/python3"
    ln -sf "$name" "$DEPS_DIR/bin/python"
}

portable_python_can_run() {
    "$DEPS_DIR/bin/python3" -c "" >/dev/null 2>&1
}

install_python_packages() {
    if portable_python_can_run; then
        "$DEPS_DIR/bin/python3" -m pip install "$@"
        return
    fi
    command -v uv >/dev/null || { echo "uv is required to prepare a cross-platform runtime" >&2; return 1; }
    uv pip install \
        --prefix "$DEPS_DIR" \
        --python-version "$PYTHON_VERSION" \
        --python-platform "$ARCH" \
        "$@"
}

install_nemo_gym_deps() {
    echo "Installing NeMo-Gym deps from $NEMO_GYM_ROOT"
    install_python_packages "$NEMO_GYM_ROOT"
}

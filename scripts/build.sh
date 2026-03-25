#!/usr/bin/env bash
# Build DontTalk.app with PyInstaller and sign for macOS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_DIR/dist"
APP_PATH="$DIST_DIR/DontTalk.app"

# Resolve signing identity: env var > auto-detect > ad-hoc
if [ -n "${CODESIGN_IDENTITY:-}" ]; then
    SIGN_IDENTITY="$CODESIGN_IDENTITY"
    SIGN_MODE="identity"
else
    SIGN_IDENTITY=$(security find-identity -p codesigning \
        | grep -m1 ')' | awk '{print $2}' || true)
    if [ -n "$SIGN_IDENTITY" ]; then
        SIGN_MODE="identity"
    else
        echo "WARNING: No codesigning identity found, falling back to ad-hoc signing."
        SIGN_MODE="adhoc"
    fi
fi

cd "$PROJECT_DIR"

echo "==> Syncing dependencies..."
uv sync

if [ "${1:-}" = "--fast" ]; then
    echo "==> Fast build (incremental)..."
    CLEAN_FLAG=""
else
    echo "==> Cleaning previous build..."
    rm -rf build "$APP_PATH" "$DIST_DIR/DontTalk"
    find "$PROJECT_DIR/donttalk" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    CLEAN_FLAG="--clean"
fi

echo "==> Running PyInstaller..."
uv run pyinstaller DontTalk.spec $CLEAN_FLAG --noconfirm

# Ensure metallib is adjacent to EVERY libmlx.dylib in the bundle
METALLIB=$(find "$APP_PATH/Contents" -name "mlx.metallib" -print -quit 2>/dev/null)
if [ -n "$METALLIB" ]; then
    while IFS= read -r dylib; do
        target_dir=$(dirname "$dylib")
        if [ ! -f "$target_dir/mlx.metallib" ]; then
            cp "$METALLIB" "$target_dir/mlx.metallib"
            echo "==> Copied metallib to $target_dir/"
        fi
    done < <(find "$APP_PATH/Contents" -name "libmlx.dylib")
fi

if [ "$SIGN_MODE" = "identity" ]; then
    echo "==> Signing app bundle (identity: $SIGN_IDENTITY)..."
    codesign --force --deep --sign "$SIGN_IDENTITY" "$APP_PATH"
else
    echo "==> Signing app bundle (ad-hoc)..."
    codesign --force --deep --sign - "$APP_PATH"
fi

echo "==> Verifying signature..."
codesign --verify --verbose "$APP_PATH"

APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
echo ""
echo "==> Build complete: $APP_PATH ($APP_SIZE)"

if [ "${1:-}" = "--fast" ]; then
    BUNDLE_ID="com.noahlyons.donttalk"
    OLD_PID=$(lsappinfo info -only pid -app "$BUNDLE_ID" 2>/dev/null | grep -o '[0-9]*' || true)
    if [ -n "$OLD_PID" ]; then
        echo "==> Stopping old instance (pid=$OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        while kill -0 "$OLD_PID" 2>/dev/null; do sleep 0.2; done
    fi
    echo "==> Launching $APP_PATH..."
    open "$APP_PATH"
else
    echo "    Run with: open $APP_PATH"
fi

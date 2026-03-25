#!/usr/bin/env bash
# Package DontTalk.app into a DMG installer.
# Usage: ./scripts/build-dmg.sh [version]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_DIR/dist"
APP_PATH="$DIST_DIR/DontTalk.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found. Run './scripts/build.sh' first."
    exit 1
fi

# Read version from pyproject.toml
PYPROJECT_VERSION=$(python3 -c "
import tomllib
with open('$PROJECT_DIR/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")

VERSION="${1:-$PYPROJECT_VERSION}"

if [ "$VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "ERROR: Version mismatch! Requested: $VERSION, pyproject.toml: $PYPROJECT_VERSION"
    exit 1
fi

DMG_PATH="$DIST_DIR/DontTalk-${VERSION}-arm64.dmg"

cd "$PROJECT_DIR"

echo "==> Creating DMG for DontTalk v${VERSION}..."

rm -f "$DMG_PATH"
create-dmg \
    --volname "DontTalk" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 128 \
    --icon "DontTalk.app" 175 190 \
    --app-drop-link 425 190 \
    --hide-extension "DontTalk.app" \
    --no-internet-enable \
    "$DMG_PATH" \
    "$APP_PATH"

APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
echo ""
echo "==> DMG complete!"
echo "    App: $APP_PATH ($APP_SIZE)"
echo "    DMG: $DMG_PATH ($DMG_SIZE)"

#!/usr/bin/env bash
# demo-convert.sh — Convert a screen recording (.mov) into README-ready assets.
#
# Produces:
#   1. A high-quality MP4 (primary — full frame rate, audio, GitHub <video> embed)
#   2. An optional GIF fallback (--gif flag, lower frame rate, no audio)
#
# Usage:
#   ./scripts/demo-convert.sh recording.mov                        # MP4 only, full res
#   ./scripts/demo-convert.sh recording.mov --crop-menu            # crop menubar
#   ./scripts/demo-convert.sh recording.mov --start 1.5 --duration 8
#   ./scripts/demo-convert.sh recording.mov --gif                  # also produce GIF
#   ./scripts/demo-convert.sh recording.mov --gif --gif-width 720  # smaller GIF

set -euo pipefail

usage() {
    echo "Usage: $0 <input.mov> [options]"
    echo ""
    echo "Options:"
    echo "  --crop-menu        Crop top 38px (macOS menubar)"
    echo "  --crop-top N       Crop top N pixels"
    echo "  --width N          MP4 width (default: native)"
    echo "  --crf N            MP4 quality, lower=better (default: 18)"
    echo "  --start N          Start time in seconds"
    echo "  --duration N       Duration in seconds"
    echo "  --gif              Also produce a GIF"
    echo "  --gif-width N      GIF width (default: 960)"
    echo "  --gif-fps N        GIF frame rate (default: 30)"
    echo "  --no-mp4           Skip MP4, only produce GIF"
    echo "  -o, --output DIR   Output directory (default: same as input)"
    exit 1
}

[[ $# -lt 1 ]] && usage

INPUT="$1"; shift
[[ ! -f "$INPUT" ]] && echo "Error: '$INPUT' not found" && exit 1

# Defaults
WIDTH=""
CRF=18
CROP_TOP=0
START=""
DURATION=""
MAKE_GIF=false
MAKE_MP4=true
GIF_WIDTH=960
GIF_FPS=30
OUTPUT_DIR="$(dirname "$INPUT")"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --crop-menu)   CROP_TOP=38; shift ;;
        --crop-top)    CROP_TOP="$2"; shift 2 ;;
        --width)       WIDTH="$2"; shift 2 ;;
        --crf)         CRF="$2"; shift 2 ;;
        --start)       START="$2"; shift 2 ;;
        --duration)    DURATION="$2"; shift 2 ;;
        --gif)         MAKE_GIF=true; shift ;;
        --gif-width)   GIF_WIDTH="$2"; MAKE_GIF=true; shift 2 ;;
        --gif-fps)     GIF_FPS="$2"; MAKE_GIF=true; shift 2 ;;
        --no-mp4)      MAKE_MP4=false; MAKE_GIF=true; shift ;;
        -o|--output)   OUTPUT_DIR="$2"; shift 2 ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

BASENAME="$(basename "${INPUT%.*}")"
MP4_OUT="$OUTPUT_DIR/${BASENAME}-demo.mp4"
GIF_OUT="$OUTPUT_DIR/${BASENAME}-demo.gif"

mkdir -p "$OUTPUT_DIR"

# Build filter chains
TRIM_ARGS=""
[[ -n "$START" ]]    && TRIM_ARGS="$TRIM_ARGS -ss $START"
[[ -n "$DURATION" ]] && TRIM_ARGS="$TRIM_ARGS -t $DURATION"

CROP_FILTER=""
if [[ "$CROP_TOP" -gt 0 ]]; then
    CROP_FILTER="crop=iw:ih-${CROP_TOP}:0:${CROP_TOP},"
fi

# --- MP4 ---
if $MAKE_MP4; then
    echo "→ Creating MP4..."
    SCALE_FILTER=""
    if [[ -n "$WIDTH" ]]; then
        SCALE_FILTER="scale=${WIDTH}:-2,"
    fi
    ffmpeg -y $TRIM_ARGS -i "$INPUT" \
        -vf "${CROP_FILTER}${SCALE_FILTER}setsar=1" \
        -c:v libx264 -preset slow -crf "$CRF" \
        -r 60 \
        -c:a aac -b:a 128k \
        -movflags +faststart \
        -pix_fmt yuv420p \
        "$MP4_OUT" 2>/dev/null

    MP4_SIZE="$(du -h "$MP4_OUT" | cut -f1 | xargs)"
    echo "  ✓ $MP4_OUT ($MP4_SIZE)"
fi

# --- GIF (optional, --gif flag) ---
if $MAKE_GIF; then
    echo "→ Creating GIF (this takes a moment)..."

    PALETTE="$(mktemp /tmp/palette-XXXXX.png)"
    FILTERS="${CROP_FILTER}scale=${GIF_WIDTH}:-2:flags=lanczos,fps=${GIF_FPS}"

    ffmpeg -y $TRIM_ARGS -i "$INPUT" \
        -vf "${FILTERS},palettegen=max_colors=256:stats_mode=diff" \
        "$PALETTE" 2>/dev/null

    ffmpeg -y $TRIM_ARGS -i "$INPUT" -i "$PALETTE" \
        -lavfi "${FILTERS} [x]; [x][1:v] paletteuse=dither=floyd_steinberg" \
        "$GIF_OUT" 2>/dev/null

    rm -f "$PALETTE"

    GIF_SIZE="$(du -h "$GIF_OUT" | cut -f1 | xargs)"
    echo "  ✓ $GIF_OUT ($GIF_SIZE)"

    GIF_BYTES="$(stat -f%z "$GIF_OUT" 2>/dev/null || stat -c%s "$GIF_OUT" 2>/dev/null)"
    if [[ "$GIF_BYTES" -gt 10485760 ]]; then
        echo ""
        echo "  ⚠ GIF is over 10MB — may not render on GitHub."
        echo "    Try: --gif-fps 20, --gif-width 720, or shorter --duration"
    fi
fi

echo ""
echo "Done."
if $MAKE_MP4; then
    echo ""
    echo "For README (GitHub video embed):"
    echo '  <video src="https://github.com/user-attachments/assets/UPLOAD_ID.mp4" width="100%"></video>'
    echo ""
    echo "  Upload: drag the MP4 into any GitHub issue/PR comment to get the URL."
fi
if $MAKE_GIF; then
    echo ""
    echo "For README (GIF fallback):"
    echo '  ![demo](./'"${BASENAME}"'-demo.gif)'
fi

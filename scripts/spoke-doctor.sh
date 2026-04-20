#!/bin/bash
# spoke-doctor.sh — check health of the spoke service fleet
#
# Pings each expected service endpoint and reports what's up, what's down,
# and what URL it tried. No repair, just diagnosis.
#
# Usage:
#   ./scripts/spoke-doctor.sh
#   ./scripts/spoke-doctor.sh --verbose

set -uo pipefail
# Note: `set -e` intentionally omitted — check_service returns non-zero
# for down services, and we want to keep checking the rest of the fleet.

VERBOSE=false
if [[ "${1:-}" == "--verbose" || "${1:-}" == "-v" ]]; then
    VERBOSE=true
fi

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    GREEN='' RED='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

UP=0
DOWN=0
OPTIONAL_DOWN=0

check_service() {
    local name="$1"
    local url="$2"
    local endpoint="$3"
    local required="${4:-true}"
    local full_url="${url}${endpoint}"

    printf "${BOLD}%-24s${RESET} %s " "$name" "$full_url"

    # 2-second timeout, follow redirects
    # 2xx/3xx = up, 401/403 = up but needs auth, anything else = error
    if response=$(curl -sS --connect-timeout 2 --max-time 3 -o /dev/null -w "%{http_code}" "$full_url" 2>/dev/null); then
        if [[ "$response" =~ ^[23] ]]; then
            printf "${GREEN}UP${RESET} (HTTP %s)\n" "$response"
            UP=$((UP + 1))
            if $VERBOSE; then
                detail=$(curl -sS --max-time 2 "$full_url" 2>/dev/null | head -c 200)
                if [[ -n "$detail" ]]; then
                    printf "  └─ %s\n" "$detail"
                fi
            fi
            return 0
        elif [[ "$response" == "401" || "$response" == "403" ]]; then
            printf "${GREEN}UP${RESET} (HTTP %s — auth required)\n" "$response"
            UP=$((UP + 1))
            return 0
        else
            printf "${RED}ERROR${RESET} (HTTP %s)\n" "$response"
            if [[ "$required" == "true" ]]; then
                DOWN=$((DOWN + 1))
            else
                OPTIONAL_DOWN=$((OPTIONAL_DOWN + 1))
            fi
            return 1
        fi
    else
        if [[ "$required" == "true" ]]; then
            printf "${RED}DOWN${RESET}\n"
            DOWN=$((DOWN + 1))
        else
            printf "${YELLOW}DOWN (optional)${RESET}\n"
            OPTIONAL_DOWN=$((OPTIONAL_DOWN + 1))
        fi
        return 1
    fi
}

resolve_url() {
    local env_var="$1"
    local default_host="$2"
    local default_port="$3"
    local val="${!env_var:-}"
    if [[ -n "$val" ]]; then
        # Strip trailing slash
        echo "${val%/}"
    else
        echo "http://${default_host}:${default_port}"
    fi
}

echo ""
printf "${BOLD}Spoke Service Fleet Health${RESET}\n"
echo "─────────────────────────────────────────────────────"
echo ""

# --- Grapheus (command proxy) ---
command_url=$(resolve_url SPOKE_COMMAND_URL localhost 8090)
check_service "Grapheus (commands)" "$command_url" "/v1/models" true

# --- Local OMLX upstream ---
omlx_upstream_url=$(resolve_url OMLX_UPSTREAM_URL localhost 8001)
check_service "OMLX upstream" "$omlx_upstream_url" "/v1/models" false

# --- Narrator ---
narrator_url="${SPOKE_NARRATOR_URL:-}"
if [[ -n "$narrator_url" ]]; then
    narrator_url="${narrator_url%/}"
    check_service "Narrator" "$narrator_url" "/v1/models" false
else
    # Narrator falls back to command URL — already checked above
    printf "${BOLD}%-24s${RESET} (using Grapheus command path)\n" "Narrator"
fi

# --- MLX-audio sidecar (TTS) ---
tts_url=$(resolve_url SPOKE_TTS_URL MacBook-Pro-2.local 9001)
check_service "MLX-audio (TTS)" "$tts_url" "/v1/models" true

# --- Whisper (remote, optional) ---
whisper_url="${SPOKE_WHISPER_URL:-}"
if [[ -n "$whisper_url" ]]; then
    whisper_url="${whisper_url%/}"
    check_service "Whisper (remote)" "$whisper_url" "/v1/models" false
else
    # Check the conventional location anyway
    check_service "Whisper (nlm2pr)" "http://nlm2pr.local:7001" "/v1/models" false
fi

# --- Spoke process itself ---
echo ""
printf "${BOLD}%-24s${RESET} " "Spoke process"
if pgrep -f "spoke" > /dev/null 2>&1; then
    pid=$(pgrep -f "spoke.__main__" 2>/dev/null | head -1 || pgrep -f "spoke" | head -1)
    printf "${GREEN}RUNNING${RESET} (PID %s)\n" "$pid"
else
    printf "${YELLOW}NOT RUNNING${RESET}\n"
fi

# --- Summary ---
echo ""
echo "─────────────────────────────────────────────────────"
TOTAL=$((UP + DOWN + OPTIONAL_DOWN))
if [[ $DOWN -eq 0 ]]; then
    if [[ $OPTIONAL_DOWN -eq 0 ]]; then
        printf "${GREEN}All %d services up.${RESET}\n" "$UP"
    else
        printf "${GREEN}%d required services up.${RESET} ${YELLOW}%d optional services down.${RESET}\n" "$UP" "$OPTIONAL_DOWN"
    fi
else
    printf "${RED}%d required services down.${RESET} %d up." "$DOWN" "$UP"
    if [[ $OPTIONAL_DOWN -gt 0 ]]; then
        printf " ${YELLOW}%d optional down.${RESET}" "$OPTIONAL_DOWN"
    fi
    echo ""
fi
echo ""

# Exit code: non-zero if any required service is down
[[ $DOWN -eq 0 ]]

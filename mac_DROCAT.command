#!/usr/bin/env bash
# Double-click / terminal launcher for macOS and Linux.
# Self-healing: prepares the versioned environment on first run (via
# archive/install/install.sh), repairs it when inconsistent, resolves port
# conflicts interactively, and launches the web UI.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/archive/install/install.sh"

clear 2>/dev/null || true
printf '%s\n' "DROCAT - Drosophila Connectome Analysis Toolkit"
printf '%s\n' "Preparing the versioned environment and launching the UI..."
printf '%s\n' ""

main() {
    DROCAT_VERSION=""
    if [[ -f "$SCRIPT_DIR/ui/config.py" ]]; then
        DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)"
    fi
    if [[ -z "$DROCAT_VERSION" ]]; then
        DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/pyproject.toml" | head -1)"
    fi
    DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
    ENV_BASE="drocat-${DROCAT_VERSION}"
    # config.json ships clean on GitHub (committed defaults); the gitignored
    # config_local.json is the per-user override and wins per key.
    CONFIG_FILE="$SCRIPT_DIR/config.json"
    CONFIG_LOCAL="$SCRIPT_DIR/config_local.json"

    # Minimal JSON reader for config.json (string values only, one level of
    # section objects; see config.json for the exact format). Commas
    # are normalized to newlines first so pretty-printed and single-line
    # JSON both work.
    json_value() {
        # $1 = section ("envs"|"tokens"), $2 = key, $3 = file
        tr ',' '\n' < "$3" | awk -v section="$1" -v key="$2" '
            {
                line = $0
                sub(/^[[:space:]]*/, "", line)
                sub(/[[:space:]]*$/, "", line)
                if (!in_section) {
                    if (line ~ "\"" section "\"") {
                        in_section = 1
                        sub(/^.*\"" section "\"/, "", line)
                    } else {
                        next
                    }
                }
                if (line ~ "\"" key "\":") {
                    rest = substr(line, index(line, "\"" key "\":") + length("\"" key "\":"))
                    sub(/^[[:space:]]*/, "", rest)
                    gsub(/[{}]/, "", rest)
                    # Stripping the closing brace can leave trailing whitespace
                    # that defeats the quote trim below; trim again first.
                    sub(/[[:space:]]*$/, "", rest)
                    gsub(/^"|"$/, "", rest)
                    print rest
                    exit
                }
                if (line ~ /^[[:space:]]*}/) { in_section = 0 }
            }
        '
    }

    # Diagnostic for the envs section: prints the version key of the first
    # envs entry that is not the current release, if any. Catches edits
    # where the version key was changed instead of the value.
    env_override_for_other_version() {
        # $1 = current version, remaining args = config files
        local version="$1" cfg other
        shift
        for cfg in "$@"; do
            [[ -f "$cfg" ]] || continue
            other="$(tr ',' '\n' < "$cfg" | awk -v ver="$version" '
                {
                    line = $0
                    sub(/^[[:space:]]*/, "", line)
                    sub(/[[:space:]]*$/, "", line)
                    if (!in_envs) {
                        if (line ~ /"envs"/) { in_envs = 1; sub(/^.*"envs"/, "", line) } else { next }
                    }
                    if (line ~ /"[^"]*"[[:space:]]*:[[:space:]]*"[^"]*"/) {
                        if (match(line, /"[^"]*"/)) {
                            key = substr(line, RSTART + 1, RLENGTH - 2)
                        }
                        if (key != ver) { print key; exit }
                    }
                    if (line ~ /^[[:space:]]*}/) { in_envs = 0 }
                }
            ' || true)"
            if [[ -n "$other" ]]; then
                printf '%s\n' "$other"
                return 0
            fi
        done
        return 1
    }

    # Update envs.<version> in the LOCAL config (config_local.json) with the
    # environment actually used, so an auto-created env is pinned for later
    # runs (it only applies while the config.json entry stays empty). The
    # committed config.json is never rewritten.
    update_config_env() {
        # $1 = version, $2 = env name, $3 = file
        [[ -f "$3" ]] || return 0
        local tmp
        tmp="$(mktemp "${TMPDIR:-/tmp}/drocat-config.XXXXXX")"
        awk -v version="$1" -v envname="$2" '
            BEGIN { in_envs = 0; done = 0 }
            {
                line = $0
                if (!done && line ~ /"envs"/) { in_envs = 1 }
                pattern = "\"" version "\"[[:space:]]*:[[:space:]]*\"[^\"]*\""
                if (in_envs && !done && line ~ pattern) {
                    sub(pattern, "\"" version "\": \"" envname "\"", line)
                    done = 1
                    in_envs = 0
                }
                print line
            }
        ' "$3" > "$tmp" && mv "$tmp" "$3"
        chmod 600 "$3"
    }

    # Version-specific custom env override: envs.<version> is only consulted
    # for the CURRENT release, so upgrading DROCAT never reuses an older
    # release's custom environment. config.json wins per key - it is the file
    # a GitHub-pulled copy edits directly; the gitignored config_local.json
    # is the developer-specific fallback for empty entries. An empty value
    # means default auto-find.
    ENV_OVERRIDE=""
    ENV_SOURCE=""
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            ENV_OVERRIDE="$(json_value envs "$DROCAT_VERSION" "$cfg" || true)"
            ENV_OVERRIDE="$(printf '%s' "$ENV_OVERRIDE" | tr -d '[:space:]')"
            if [[ -n "$ENV_OVERRIDE" ]]; then
                ENV_SOURCE="$(basename "$cfg")"
                break
            fi
        fi
    done
    # Hint when an envs entry exists for another release: the version key
    # must match the release being launched (easy to mis-edit).
    if [[ -z "$ENV_OVERRIDE" ]]; then
        other_version="$(env_override_for_other_version "$DROCAT_VERSION" "$CONFIG_FILE" "$CONFIG_LOCAL" || true)"
        if [[ -n "$other_version" ]]; then
            printf '%s\n' "Note: configs set an env for release $other_version, but this build resolves envs.$DROCAT_VERSION - the envs key must match the release. Set envs.$DROCAT_VERSION to pin an env for this release." >&2
        fi
    fi

    find_conda() {
        if command -v conda >/dev/null 2>&1; then
            command -v conda
            return
        fi
        local candidate
        for candidate in \
            "$HOME/miniconda3/bin/conda" \
            "$HOME/anaconda3/bin/conda" \
            "$HOME/miniforge3/bin/conda" \
            "/opt/miniconda3/bin/conda" \
            "/opt/anaconda3/bin/conda" \
            "/usr/local/miniconda3/bin/conda" \
            "/usr/local/anaconda3/bin/conda"; do
            [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
        done
    }

    # Resolve the Python binary inside a named conda env without going through
    # `conda run`.  `conda run` allocates a pseudo-TTY, which fails when macOS
    # exhausts its PTY pool (kern.tty.ptmx_max, default 255/511) with
    # `[forkpty: Device not configured]` / `[Could not create a new process and
    # open a pseudo-tty.]`, so the env's python is invoked directly instead
    # (it imports the env's site-packages identically).  The conda base is
    # resolved via `conda info --base` (non-interactive, never allocates a PTY),
    # with a prefix-derivation fallback for non-standard/symlinked conda paths.
    env_bin_python() {
        # $1 = env name; prints the env's python path when it exists, else nothing.
        local base py
        base="$("$CONDA_BIN" info --base 2>/dev/null || true)"
        if [[ -z "$base" ]]; then
            base="$(dirname "$(dirname "$CONDA_BIN")")"
        fi
        py="$base/envs/$1/bin/python"
        if [[ -x "$py" ]]; then
            printf '%s\n' "$py"
        fi
        return 0
    }

    env_exists() {
        # $1 = env name. Pure filesystem check (no conda subprocess, so no PTY
        # is allocated); the env's python path is authoritative for existence.
        local py
        py="$(env_bin_python "$1")"
        [[ -n "$py" ]]
    }

    resolve_env() {
        local index candidate py
        ENV_NAME=""
        if [[ -n "$ENV_OVERRIDE" ]]; then
            py="$(env_bin_python "$ENV_OVERRIDE")"
            if [[ -n "$py" ]] && "$py" -c \
                'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
                >/dev/null 2>&1; then
                ENV_NAME="$ENV_OVERRIDE"
                printf 'Using custom environment %s (set in %s).\n' "$ENV_NAME" "$ENV_SOURCE"
                # config_local.json is the developer-specific fallback: it
                # only fills entries that are empty in config.json.
                if [[ "$ENV_SOURCE" == "config_local.json" && -f "$CONFIG_FILE" ]]; then
                    printf '%s\n' "Note: config.json has no envs.$DROCAT_VERSION value; using the config_local.json fallback." >&2
                fi
                return 0
            fi
            if env_exists "$ENV_OVERRIDE"; then
                printf '%s\n' "ERROR: environment '$ENV_OVERRIDE' (custom env from $ENV_SOURCE) exists but is not Python 3.11." >&2
                printf '%s\n' "Fix it, remove it, or clear the envs.$DROCAT_VERSION entry; DROCAT never silently switches environments." >&2
                return 1
            fi
            # The env does not exist yet: leave ENV_NAME empty so the caller
            # runs the installer, which creates it and installs dependencies.
            return 0
        fi
        for index in $(seq 0 20); do
            if [[ "$index" -eq 0 ]]; then
                candidate="$ENV_BASE"
            else
                candidate="${ENV_BASE}-$((index + 1))"
            fi
            py="$(env_bin_python "$candidate")"
            if [[ -n "$py" ]] && "$py" -c \
                'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
                >/dev/null 2>&1; then
                ENV_NAME="$candidate"
                printf 'Using environment %s (auto-detected).\n' "$ENV_NAME"
                return 0
            fi
        done
        return 0
    }

    CONDA_BIN="$(find_conda || true)"
    if [[ -z "$CONDA_BIN" ]]; then
        printf '%s\n' "Conda is not installed; running the one-click installer."
        "$INSTALLER"
        CONDA_BIN="$(find_conda || true)"
    fi
    [[ -n "$CONDA_BIN" ]] || { printf '%s\n' "ERROR: Conda installation failed." >&2; return 1; }

    resolve_env || return 1
    if [[ -z "$ENV_NAME" ]]; then
        if [[ -n "$ENV_OVERRIDE" ]]; then
            printf '%s\n' "Environment '$ENV_OVERRIDE' (custom env from $ENV_SOURCE) does not exist; running the one-click installer to create it."
        else
            printf '%s\n' "Environment $ENV_BASE not found; running the one-click installer."
        fi
        "$INSTALLER"
        resolve_env || return 1
    fi
    [[ -n "$ENV_NAME" ]] || { printf '%s\n' "ERROR: no usable $ENV_BASE environment was found." >&2; return 1; }

    # Persist the resolved environment into the LOCAL config when no custom
    # name was configured, so the auto-found env is pinned for later runs
    # (it only applies while the config.json entry stays empty). The
    # committed config.json is never rewritten.
    if [[ -z "$ENV_OVERRIDE" && -f "$CONFIG_LOCAL" ]]; then
        update_config_env "$DROCAT_VERSION" "$ENV_NAME" "$CONFIG_LOCAL"
    fi

    local py
    py="$(env_bin_python "$ENV_NAME")"
    if [[ -z "$py" ]] \
        || ! "$py" -c \
            'import nicegui, numpy, pandas, neuprint, neuronbridge_client' >/dev/null 2>&1 \
        || ! "$py" -m pip check >/dev/null 2>&1; then
        printf '%s\n' "Repairing dependencies in $ENV_NAME..."
        "$INSTALLER"
        resolve_env || return 1
    fi

    # --- Token hint -----------------------------------------------------
    # Remind users who skipped the installer prompt that tokens can be set
    # later in the UI Settings tab or in config.json (config_local.json is
    # the optional developer fallback).
    local neuprint_token=""
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            neuprint_token="$(json_value tokens neuprint "$cfg" || true)"
            neuprint_token="$(printf '%s' "$neuprint_token" | tr -d '[:space:]')"
            [[ -n "$neuprint_token" ]] && break
        fi
    done
    if [[ -z "$neuprint_token" || "$neuprint_token" == "YOUR_NEUPRINT_TOKEN_HERE" ]]; then
        printf '%s\n' "Tip: the NeuPrint token is not configured yet - set it in the UI Settings tab or in config.json (the CAVE token is optional; only needed for FlyWire FAFB online fetching)."
        printf '%s\n' "     Get a NeuPrint token from: https://neuprint.janelia.org/account"
        printf '%s\n' "     Get a CAVE token from: https://codex.flywire.ai/auth_token"
    fi

    # --- PTY-pressure warning --------------------------------------------
    # `[forkpty: Device not configured]` happens when macOS's pseudo-terminal
    # pool (kern.tty.ptmx_max, default 255/511) is exhausted, e.g. too many
    # terminals or `conda run` processes are open.  Warn before launch so the
    # user can raise the limit rather than hitting a confusing failure.
    local ptmx_max=0 pty_count=0
    ptmx_max="$(sysctl -n kern.tty.ptmx_max 2>/dev/null || printf '0')"
    pty_count="$(lsof -n 2>/dev/null | grep -c '/dev/ttys' || printf '0')"
    if [[ "$ptmx_max" =~ ^[0-9]+$ ]] && [[ "$ptmx_max" -gt 0 ]] \
        && [[ "$pty_count" =~ ^[0-9]+$ ]] && [[ "$pty_count" -ge $((ptmx_max - 25)) ]]; then
        printf '%s\n' "Warning: $pty_count of $ptmx_max pseudo-terminals are in use. If you see '[forkpty: Device not configured]', raise the PTY limit:" >&2
        printf '%s\n' "    sudo sysctl -w kern.tty.ptmx_max=1024" >&2
    fi

    # --- Port-conflict guard ---------------------------------------------
    # If the UI port is already in use, ask the user whether to start on a new
    # port, kill the existing DROCAT process and restart, or cancel. When the
    # script is not interactive (agents, CI), keep the previous automatic
    # behavior: open the browser if a DROCAT instance owns the port, otherwise
    # fail with a hint.
    APP_PORT="${DROCAT_UI_PORT:-$(sed -n 's/^APP_PORT = \([0-9][0-9]*\)/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)}"
    APP_PORT="${APP_PORT:-8080}"

    port_in_use() {
        local port="${1:-$APP_PORT}"
        if command -v lsof >/dev/null 2>&1; then
            lsof -ti "tcp:$port" -sTCP:LISTEN >/dev/null 2>&1
        elif command -v nc >/dev/null 2>&1; then
            nc -z 127.0.0.1 "$port" >/dev/null 2>&1
        else
            (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
        fi
    }

    is_drocat_owner() {
        local pid="$1" cmd
        [[ -n "$pid" ]] || return 1
        cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        echo "$cmd" | grep -qE "ui/app\.py|drocat"
    }

    launch_ui() {
        local port="${1:-$APP_PORT}" py
        printf 'Starting DROCAT v%s in %s at http://127.0.0.1:%s...\n' "$DROCAT_VERSION" "$ENV_NAME" "$port"
        py="$(env_bin_python "$ENV_NAME")"
        if [[ -z "$py" ]]; then
            printf '%s\n' "ERROR: Python not found in environment '$ENV_NAME'." >&2
            return 1
        fi
        cd "$SCRIPT_DIR"
        export DROCAT_UI_PORT="$port"
        exec "$py" ui/app.py
    }

    if port_in_use; then
        owner_pid="$(lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
        owner_cmd="$(ps -p "$owner_pid" -o command= 2>/dev/null || true)"
        if [[ -t 0 ]]; then
            # Interactive: let the user decide what to do with the busy port.
            while true; do
                printf '\nPort %s is already in use by PID %s:\n    %s\n' "$APP_PORT" "$owner_pid" "$owner_cmd"
                printf '  [1] Start DROCAT on a new port\n'
                if is_drocat_owner "$owner_pid"; then
                    printf '  [2] Kill the existing DROCAT process and restart on port %s\n' "$APP_PORT"
                else
                    printf '  [2] Not allowed: the process on port %s is not DROCAT - stop it manually, then retry\n' "$APP_PORT"
                fi
                printf '  [3] Cancel\n'
                printf 'Your choice [1-3]: '
                read -r choice || break
                case "$choice" in
                    1)
                        new_port="$APP_PORT"
                        while port_in_use "$new_port"; do
                            new_port=$((new_port + 1))
                            [[ "$new_port" -le 65535 ]] || { printf 'No free port found.\n' >&2; return 1; }
                        done
                        printf 'Port %s is busy; starting on port %s instead.\n' "$APP_PORT" "$new_port"
                        launch_ui "$new_port"
                        ;;
                    2)
                        if is_drocat_owner "$owner_pid"; then
                            printf 'Stopping PID %s...\n' "$owner_pid"
                            kill "$owner_pid" 2>/dev/null || true
                            for _ in $(seq 1 20); do
                                port_in_use || break
                                sleep 0.5
                            done
                            if port_in_use; then
                                printf 'ERROR: the process on port %s did not stop.\n' "$APP_PORT" >&2
                                return 1
                            fi
                            launch_ui "$APP_PORT"
                        else
                            printf 'The process on port %s is not DROCAT; it will not be killed automatically.\n' "$APP_PORT"
                        fi
                        ;;
                    3)
                        printf 'Cancelled.\n'
                        return 1
                        ;;
                    *)
                        printf 'Invalid choice.\n'
                        ;;
                esac
            done
            return 1
        else
            # Non-interactive (agent, CI): previous automatic behavior.
            if is_drocat_owner "$owner_pid"; then
                printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
                printf 'Opening it in your browser...\n'
                if command -v open >/dev/null 2>&1; then
                    open "http://127.0.0.1:$APP_PORT"
                fi
                return 0
            fi
            # Otherwise probe the page to rule out a non-DROCAT server on the port.
            if curl -s --max-time 2 "http://127.0.0.1:$APP_PORT/" 2>/dev/null | grep -q "drocat-cobalt"; then
                printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
                printf 'Opening it in your browser...\n'
                if command -v open >/dev/null 2>&1; then
                    open "http://127.0.0.1:$APP_PORT"
                fi
                return 0
            fi
            printf 'ERROR: Port %s is already in use by another application (PID %s).\n' "$APP_PORT" "$owner_pid" >&2
            printf 'Stop it manually, run with DROCAT_UI_PORT=<free port>, or run interactively to choose.\n' >&2
            return 1
        fi
    fi

    printf 'Starting DROCAT v%s in %s...\n' "$DROCAT_VERSION" "$ENV_NAME"
    launch_ui "$APP_PORT"
}

main || rc=$?
if [[ "${rc:-0}" -ne 0 ]]; then
    printf '\n%s\n' "DROCAT could not start. Review the messages above."
    if [[ -t 0 ]]; then
        read -r -p "Press Return to close." _
    fi
    exit "$rc"
fi

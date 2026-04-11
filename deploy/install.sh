#!/bin/bash
# Kriya – Pi Zero / Debian / Ubuntu install script
# Supports: armv6 (Pi Zero W), armhf (Pi Zero 2 W, Pi 2/3), arm64 (Pi 4/5), x86_64
# Usage: sudo bash deploy/install.sh
set -euo pipefail

INSTALL_DIR=/usr/lib/kriya
DATA_DIR=/var/lib/kriya
LOG_DIR=/var/log/kriya
CONF_DIR=/etc/kriya
BIN=/usr/local/bin/kriya
SERVICE=/etc/systemd/system/kriya.service
USER=kriya

# ── Architecture detection ────────────────────────────────────────────────
ARCH=$(uname -m)
BITS=$(getconf LONG_BIT)

case "$ARCH" in
  armv6l)  ARCH_LABEL="ARMv6 32-bit (Pi Zero W)"         ; ARCH_WARN=1 ;;
  armv7l)  ARCH_LABEL="ARMv7 32-bit (Pi Zero 2W / Pi 3)" ; ARCH_WARN=0 ;;
  aarch64) ARCH_LABEL="ARM64 64-bit (Pi 4 / Pi 5)"       ; ARCH_WARN=0 ;;
  x86_64)  ARCH_LABEL="x86_64 64-bit"                    ; ARCH_WARN=0 ;;
  *)       ARCH_LABEL="Unknown ($ARCH)"                   ; ARCH_WARN=0 ;;
esac

# Python: prefer 3.11+ for tomllib; 3.10 works without TOML project files
PYTHON=$(command -v python3.12 2>/dev/null || \
         command -v python3.11 2>/dev/null || \
         command -v python3.10 2>/dev/null || \
         command -v python3    2>/dev/null || echo "")
PYTHON_VER=$($PYTHON --version 2>&1 | awk '{print $2}')

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║        Kriya Installer v0.2.0           ║"
echo "  ║   Raspberry Pi · Debian · Ubuntu          ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""
echo "  Architecture : $ARCH_LABEL"
echo "  Word size    : ${BITS}-bit"
echo "  Kernel       : $(uname -r)"
echo "  Python       : $PYTHON_VER ($PYTHON)"
echo ""

if [ "$EUID" -ne 0 ]; then
  echo "  ✗  Run as root: sudo bash deploy/install.sh"
  exit 1
fi

if [ -z "$PYTHON" ]; then
  echo "  ✗  Python 3.10+ not found."
  echo "     sudo apt-get install -y python3"
  exit 1
fi

PYMAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PYMINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PYMAJOR" -lt 3 ] || ([ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]); then
  echo "  ✗  Python 3.10+ required. Found: $PYTHON_VER"
  exit 1
fi

if [ "$PYMINOR" -lt 11 ]; then
  echo "  ⚠  Python $PYTHON_VER: TOML project files require 3.11+."
  echo "     Projects can still be created via CLI and API."
  echo ""
fi

if [ "${ARCH_WARN:-0}" -eq 1 ]; then
  echo "  ⚠  ARMv6 (Pi Zero W) detected:"
  echo "     • Use Raspberry Pi OS Lite 32-bit (64-bit image does NOT support ARMv6)"
  echo "     • Set KRIYA_MAX_AGENTS=1 (512 MB RAM)"
  echo "     • Ollama is NOT supported on ARMv6 — use Anthropic or OpenAI API keys"
  echo "     • Recommended model: claude-3-5-haiku or gpt-4o-mini"
  echo ""
fi

echo "  →  Creating system user: $USER"
id -u "$USER" &>/dev/null || useradd --system --no-create-home --shell /sbin/nologin "$USER"

echo "  →  Creating directories"
for d in "$DATA_DIR" "$LOG_DIR" "$CONF_DIR" \
          "$DATA_DIR/vault" "$DATA_DIR/projects" \
          "$DATA_DIR/skills" "$DATA_DIR/static"; do
  mkdir -p "$d"
done
chown -R "$USER:$USER" "$DATA_DIR" "$LOG_DIR"
chmod 700 "$DATA_DIR/vault"

echo "  →  Installing source → $INSTALL_DIR"
[ -d "$INSTALL_DIR" ] && rm -rf "$INSTALL_DIR"
cp -r "$(dirname "$0")/.." "$INSTALL_DIR"
chown -R "$USER:$USER" "$INSTALL_DIR"

echo "  →  Installing CLI → $BIN"
printf '#!/bin/bash\nexec %s /usr/lib/kriya/bin/kriya "$@"\n' "$PYTHON" > "$BIN"
chmod +x "$BIN"

if [ ! -f "$CONF_DIR/kriya.env" ]; then
  echo "  →  Creating config → $CONF_DIR/kriya.env"

  if   [ "$ARCH" = "armv6l" ]; then MAX_AGENTS=1; MEM_LIMIT="180M"; TIMEOUT=300
  elif [ "$ARCH" = "armv7l" ]; then MAX_AGENTS=2; MEM_LIMIT="320M"; TIMEOUT=180
  else                               MAX_AGENTS=4; MEM_LIMIT="512M"; TIMEOUT=120
  fi

  cat > "$CONF_DIR/kriya.env" << ENVEOF
# Kriya Environment Configuration
# Architecture: $ARCH_LABEL (${BITS}-bit)

# LLM Providers — set at least one
#ANTHROPIC_API_KEY=sk-ant-...
#OPENAI_API_KEY=sk-...

# Ollama local models (ARMv7+, ARM64, x86_64 only — NOT supported on ARMv6)
#OLLAMA_BASE_URL=http://localhost:11434
#OLLAMA_MODEL=llama3

# Daemon
KRIYA_HOST=0.0.0.0
KRIYA_PORT=7777
KRIYA_LOG_LEVEL=INFO
KRIYA_MAX_AGENTS=$MAX_AGENTS

# Vault master key — set a strong random value for portability across machines.
# If unset, Kriya auto-generates one and saves it to vault/.vault_passphrase.
# WARNING: if you move the vault to another machine, this value must match.
KRIYA_VAULT_PASS=change-me-to-a-strong-random-passphrase
ENVEOF
  chmod 600 "$CONF_DIR/kriya.env"
fi

if   [ "$ARCH" = "armv6l" ]; then MEM_LIMIT="180M"
elif [ "$ARCH" = "armv7l" ]; then MEM_LIMIT="320M"
else                               MEM_LIMIT="512M"
fi

echo "  →  Installing systemd service"
cat > "$SERVICE" << SVCEOF
[Unit]
Description=Kriya Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$DATA_DIR
ExecStart=$PYTHON $INSTALL_DIR/kriya/daemon.py
Restart=on-failure
RestartSec=5s

EnvironmentFile=-$CONF_DIR/kriya.env
Environment=KRIYA_BASE=$DATA_DIR

MemoryMax=$MEM_LIMIT
MemorySwapMax=0
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=$DATA_DIR $LOG_DIR /tmp
ProtectHome=yes
CapabilityBoundingSet=
AmbientCapabilities=
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kriya

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable kriya

echo ""
echo "  ✓  Installed!  Architecture: $ARCH_LABEL"
echo ""
echo "  1. sudo nano $CONF_DIR/kriya.env   (add API key + set VAULT_PASS)"
echo "  2. sudo systemctl start kriya"
echo "  3. agent login"
echo "  4. agent status"
echo "  5. open http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo localhost):7777"
echo ""

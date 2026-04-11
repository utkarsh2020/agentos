# Kriya — Installation Guide

Complete guide to installing Kriya on any supported platform.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Development)](#quick-start-development)
- [Raspberry Pi (Production)](#raspberry-pi-production)
  - [Choose the Right OS Image](#choose-the-right-os-image)
  - [Flash and Boot](#flash-and-boot)
  - [Install Kriya](#install-kriya)
  - [Post-Install Configuration](#post-install-configuration)
- [Manual Installation (Any Linux)](#manual-installation-any-linux)
- [macOS (Development Only)](#macos-development-only)
- [Configuration Reference](#configuration-reference)
  - [LLM Provider Setup](#llm-provider-setup)
  - [Vault & Security](#vault--security)
  - [Daemon Tuning](#daemon-tuning)
- [Verify Installation](#verify-installation)
- [Systemd Service Management](#systemd-service-management)
- [Upgrading](#upgrading)
- [Uninstalling](#uninstalling)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| **Python** | 3.10+ (3.11+ recommended) | `tomllib` (TOML project files) requires 3.11 |
| **RAM** | 512 MB | Pi Zero W baseline; 1 GB+ for parallel agents |
| **Disk** | 100 MB | Source + SQLite DB + logs |
| **LLM provider** | At least one | Anthropic API key, OpenAI API key, or local Ollama |
| **OS** | POSIX | Raspberry Pi OS, Debian, Ubuntu, macOS |
| **Architecture** | ARMv6+ / x86_64 | See [board table](#choose-the-right-os-image) |

**No external Python dependencies.** Kriya uses only the Python standard library (`sqlite3`, `asyncio`, `http.server`, `urllib.request`, `hmac`, `hashlib`, `json`, `pathlib`, etc.).

---

## Quick Start (Development)

For local development on any Linux or macOS machine:

```bash
# 1. Clone the repository
git clone https://github.com/utkarsh2020/kriya.git
cd kriya

# 2. Set at least one LLM provider key
export ANTHROPIC_API_KEY=sk-ant-...     # Anthropic Claude
# OR
export OPENAI_API_KEY=sk-...            # OpenAI
# OR
export OLLAMA_MODEL=llama3              # Local Ollama (not on ARMv6)

# 3. Start the daemon (foreground)
python3 kriya/daemon.py

# 4. In another terminal — verify
python3 bin/agent login                  # use the password printed at first boot
python3 bin/agent status

# 5. Open the dashboard
open http://localhost:7777
```

On first boot, a random admin password is generated and saved to `first_run_credentials.txt` in the data directory:

```bash
# Development
cat first_run_credentials.txt

# Production (systemd install — file is owned by the kriya system user)
sudo cat /var/lib/kriya/first_run_credentials.txt
```

The file is deleted automatically after you change the admin password. Change it immediately:

```bash
python3 bin/agent user passwd admin
```

### Run the test suite

```bash
python3 tests/test_suite.py
# Expected: 41 passed, 0 failed
```

---

## Raspberry Pi (Production)

### Choose the Right OS Image

| Board | CPU | OS Image | Ollama? |
|---|---|---|---|
| **Pi Zero W** | ARMv6 32-bit | Raspberry Pi OS Lite **32-bit** (Bookworm) | No |
| **Pi Zero 2 W** | ARMv7/ARMv8 | Raspberry Pi OS Lite 32-bit or 64-bit | Yes (q4 only) |
| **Pi 3** | ARMv7/ARMv8 | Raspberry Pi OS Lite 32-bit or 64-bit | Yes (q4 only) |
| **Pi 4 / Pi 5** | ARM64 | Raspberry Pi OS Lite **64-bit** (recommended) | Yes |

> **Pi Zero W users:** The 64-bit OS image does **not** support ARMv6. You **must** use the 32-bit image.

### Flash and Boot

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash the correct OS image (see table above) to your SD card
3. In Imager settings, enable SSH and configure WiFi
4. Boot the Pi and SSH in:

```bash
ssh pi@raspberrypi.local
```

### Install Kriya

```bash
# Transfer the repo to the Pi
scp -r kriya/ pi@raspberrypi.local:~
# OR clone directly on the Pi
ssh pi@raspberrypi.local
git clone https://github.com/utkarsh2020/kriya.git
cd kriya

# Run the automated installer (requires root)
sudo bash deploy/install.sh
```

The installer automatically:
- Detects your CPU architecture (`uname -m`)
- Sets memory limits per architecture (180 MB ARMv6, 320 MB ARMv7, 512 MB ARM64/x86_64)
- Sets agent count defaults (1 for ARMv6, 2 for ARMv7, 4 for others)
- Creates a `kriya` system user
- Installs source to `/usr/lib/kriya`
- Creates data directory at `/var/lib/kriya`
- Generates `/etc/kriya/kriya.env` config file
- Installs and enables the `kriya.service` systemd unit
- Warns if ARMv6 or missing Python 3.11

### Post-Install Configuration

**Step 1: Add your API key**

```bash
sudo nano /etc/kriya/kriya.env
```

Uncomment and set at least one provider:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
# OR
OPENAI_API_KEY=sk-...
```

**Step 2: Set the vault passphrase**

```bash
# In /etc/kriya/kriya.env, change:
KRIYA_VAULT_PASS=your-strong-random-passphrase
```

> This encrypts all secrets at rest. **Set this before first run** and do not change it afterward (existing secrets become unreadable).

**Step 3: Start the service**

```bash
sudo systemctl start kriya
sudo systemctl status kriya

# View live logs
journalctl -u kriya -f
```

**Step 4: Login and verify**

```bash
sudo cat /var/lib/kriya/first_run_credentials.txt   # get the generated admin password
agent login                   # use the password from the file above
agent status                  # should show "running"
agent user passwd admin       # change the password (deletes the credentials file)
```

**Step 5: Open the dashboard**

```
http://<pi-ip>:7777
```

Find your Pi's IP with: `hostname -I`

---

## Manual Installation (Any Linux)

If you prefer not to use the installer script:

```bash
# 1. Create system user
sudo useradd --system --no-create-home --shell /sbin/nologin kriya

# 2. Create directories
sudo mkdir -p /var/lib/kriya/{vault,projects,skills,static}
sudo mkdir -p /var/log/kriya /etc/kriya
sudo chown -R kriya:kriya /var/lib/kriya /var/log/kriya
sudo chmod 700 /var/lib/kriya/vault

# 3. Copy source
sudo cp -r . /usr/lib/kriya
sudo chown -R kriya:kriya /usr/lib/kriya

# 4. Create CLI wrapper
sudo tee /usr/local/bin/agent << 'EOF'
#!/bin/bash
exec python3 /usr/lib/kriya/bin/agent "$@"
EOF
sudo chmod +x /usr/local/bin/agent

# 5. Create config
sudo tee /etc/kriya/kriya.env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
KRIYA_HOST=0.0.0.0
KRIYA_PORT=7777
KRIYA_LOG_LEVEL=INFO
KRIYA_MAX_AGENTS=3
KRIYA_VAULT_PASS=change-me-please
EOF
sudo chmod 600 /etc/kriya/kriya.env

# 6. Create systemd service (see deploy/kriya.service for template)
sudo cp deploy/kriya.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kriya
```

---

## macOS (Development Only)

macOS is supported for development. Production deployments should use Linux.

```bash
# Ensure Python 3.11+ (via Homebrew)
brew install python@3.12

# Clone and run
git clone https://github.com/utkarsh2020/kriya.git
cd kriya
export ANTHROPIC_API_KEY=sk-ant-...
python3 kriya/daemon.py
```

> Note: macOS symlinks `/tmp` to `/private/tmp`. Kriya handles this automatically.

---

## Configuration Reference

All configuration is via environment variables (or `/etc/kriya/kriya.env` on production installs).

### LLM Provider Setup

You need at least one provider configured. The auto-fallback order is: Anthropic → OpenAI → Ollama.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Default Claude model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Override for compatible APIs (Groq, Together, etc.) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | — | **Must be set** to enable Ollama (e.g. `llama3`) |

**Per-device model recommendations:**

| Device | Recommended Model | Ollama? |
|---|---|---|
| Pi Zero W (ARMv6) | `claude-3-5-haiku` or `gpt-4o-mini` | Not supported |
| Pi Zero 2 W (ARMv7) | `claude-3-5-haiku` or `gpt-4o-mini` | q4 models only |
| Pi 4 / Pi 5 (ARM64) | Any | Full support |
| x86_64 server | Any | Full support |

### Vault & Security

| Variable | Default | Description |
|---|---|---|
| `KRIYA_VAULT_PASS` | auto-generated (saved to `vault/.vault_passphrase`) | Master key for secrets vault — set explicitly in production for portability across machines |
| `KRIYA_JWT_SECRET` | auto-persisted to `.jwt_secret` | HS256 signing key — survives restarts automatically; set explicitly in production |
| `KRIYA_CORS_ORIGINS` | _(none — CORS disabled)_ | Comma-separated allowed origins for the API. Use `*` only for public APIs. |

### Daemon Tuning

| Variable | Default | Description |
|---|---|---|
| `KRIYA_BASE` | source directory | Base path for DB, vault, logs, projects |
| `KRIYA_HOST` | `0.0.0.0` | API bind address |
| `KRIYA_PORT` | `7777` | API port |
| `KRIYA_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `KRIYA_MAX_AGENTS` | `3` | Max concurrent agents (1 for Pi Zero W) |

Optional TOML config at `$KRIYA_BASE/kriya.toml`:

```toml
[daemon]
host                  = "0.0.0.0"
port                  = 7777
log_level             = "INFO"
max_concurrent_agents = 2
agent_timeout_sec     = 300
short_term_capacity   = 50
```

---

## Verify Installation

Run these checks after installation:

```bash
# 1. Service is running
sudo systemctl status kriya
# Should show: active (running)

# 2. CLI connects
agent status
# Should show: version, arch, uptime, providers

# 3. API responds
curl -s http://localhost:7777/api/health
# Should return: {"ok": true}

# 4. Login works
sudo cat /var/lib/kriya/first_run_credentials.txt   # get the generated password
agent login

# 5. Dashboard loads
curl -s -o /dev/null -w "%{http_code}" http://localhost:7777/
# Should return: 200

# 6. Run a test project (if you have an API key set)
agent run examples/newsletter.toml --follow
```

---

## Systemd Service Management

```bash
sudo systemctl start   kriya    # Start the daemon
sudo systemctl stop    kriya    # Stop the daemon
sudo systemctl restart kriya    # Restart after config changes
sudo systemctl enable  kriya    # Start on boot
sudo systemctl disable kriya    # Don't start on boot

# View logs
journalctl -u kriya -f          # Follow live
journalctl -u kriya --since today
journalctl -u kriya -n 100      # Last 100 lines
```

---

## Upgrading

```bash
# 1. Stop the service
sudo systemctl stop kriya

# 2. Pull / copy new source
cd /usr/lib/kriya
sudo git pull origin main
# OR: overwrite with new release tarball

# 3. Restart
sudo systemctl start kriya
agent status    # verify new version
```

Your database (`/var/lib/kriya/kriya.db`) and vault (`/var/lib/kriya/vault/`) are preserved — they live outside the source directory.

---

## Uninstalling

```bash
sudo systemctl stop kriya
sudo systemctl disable kriya
sudo rm /etc/systemd/system/kriya.service
sudo systemctl daemon-reload

sudo rm -rf /usr/lib/kriya
sudo rm /usr/local/bin/agent

# Optional: remove data (DESTRUCTIVE — deletes DB, vault, projects)
sudo rm -rf /var/lib/kriya /var/log/kriya /etc/kriya
sudo userdel kriya
```

---

## Troubleshooting

### "Python 3.10+ not found"

Install Python 3.11+ from your package manager:

```bash
# Debian / Raspberry Pi OS
sudo apt-get update && sudo apt-get install -y python3

# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y python3.12
```

### "TOML requires Python 3.11+"

Python 3.10 works but doesn't have `tomllib`. You can still create projects via the API and CLI — you just can't load `.toml` project files. Upgrade to Python 3.11+ to use TOML.

### Dashboard won't load

1. Check the daemon is running: `sudo systemctl status kriya`
2. Check the port: `curl http://localhost:7777/api/health`
3. Check firewall: `sudo ufw allow 7777/tcp` (if using ufw)
4. Check Pi IP: `hostname -I`

### "Unauthorized" on API calls

Your JWT token expired (1-hour TTL). Re-login:

```bash
agent login
```

### Agent runs fail with "No providers configured"

At least one LLM provider must be set:

```bash
# Check config
cat /etc/kriya/kriya.env | grep -E "(ANTHROPIC|OPENAI|OLLAMA)"

# Test API key
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-5-haiku-20241022","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
```

### Pi Zero W runs out of memory

- Set `KRIYA_MAX_AGENTS=1` in config
- Use lightweight models (`claude-3-5-haiku`, `gpt-4o-mini`)
- Don't run Ollama on ARMv6 (not supported)
- The systemd unit caps memory at 180 MB for ARMv6

### Vault decryption errors

If you changed `KRIYA_VAULT_PASS` after creating secrets, old secrets cannot be decrypted. Either:
1. Restore the original passphrase
2. Delete `vault/master.key` and re-create all secrets (vault data is lost)

```bash
# Nuclear option: reset vault
sudo systemctl stop kriya
sudo rm -rf /var/lib/kriya/vault
sudo systemctl start kriya
# Re-create secrets via CLI or dashboard
```

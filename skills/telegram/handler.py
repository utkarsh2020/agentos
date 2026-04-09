"""
Telegram Bot API skill for AgentOS.

Send messages via Telegram Bot API.
https://core.telegram.org/bots/api

Secrets required (via vault or env):
- TELEGRAM_BOT_TOKEN: Bot token from @BotFather
- TELEGRAM_CHAT_ID: Target chat ID to send to

Params:
- chat_id: Override default chat ID
- text: Message text to send
- parse_mode: "Markdown" or "HTML" for formatting
"""
import json
import urllib.request
import urllib.error

SKILL_ID = "telegram.send"


def handle(params: dict, secrets: dict) -> dict:
    """
    Send a message via Telegram Bot API.
    
    Args:
        params: {chat_id?, text, parse_mode?}
        secrets: {TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID?}
    
    Returns:
        {"ok": bool, "message_id": int, "chat": dict}
    """
    # Get secrets - check vault first, then env vars
    token = (secrets.get("TELEGRAM_BOT_TOKEN") or 
             params.get("token") or
             "")
    chat_id = (secrets.get("TELEGRAM_CHAT_ID") or 
               params.get("chat_id") or
               "")

    # Get message params
    text = params.get("text", "")
    parse_mode = params.get("parse_mode", "")  # "Markdown" or "HTML"

    # Validation
    if not token:
        return {"error": "TELEGRAM_BOT_TOKEN required (set in vault or pass as token param)"}
    if not chat_id:
        return {"error": "TELEGRAM_CHAT_ID required (set in vault or pass as chat_id param)"}
    if not text:
        return {"error": "text param required"}

    # Build request
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                return {
                    "ok": True,
                    "message_id": result.get("result", {}).get("message_id"),
                    "chat": result.get("result", {}).get("chat"),
                }
            else:
                return {
                    "ok": False,
                    "error_code": result.get("error_code"),
                    "description": result.get("description"),
                }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            return {"error": f"HTTP {e.code}", "details": error_body}
        except Exception:
            return {"error": f"HTTP {e.code}", "body": e.read().decode()[:512]}
    except Exception as e:
        return {"error": str(e)}


# Optional: also expose send_photo, send_document, etc. in same handler
def handle_photo(params: dict, secrets: dict) -> dict:
    """Send a photo via Telegram Bot API."""
    token = secrets.get("TELEGRAM_BOT_TOKEN") or params.get("token")
    chat_id = secrets.get("TELEGRAM_CHAT_ID") or params.get("chat_id")
    photo = params.get("photo", "")  # URL or file_id
    caption = params.get("caption", "")
    
    if not token or not chat_id or not photo:
        return {"error": "TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and photo required"}
    
    api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    payload = {"chat_id": chat_id, "photo": photo}
    if caption:
        payload["caption"] = caption
        
    data = json.dumps(payload).encode()
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def handle_get_updates(params: dict, secrets: dict) -> dict:
    """Poll for incoming messages via Telegram Bot API getUpdates.
    
    This enables the bot to receive incoming messages (trigger-on-message).
    Uses long polling - waits up to timeout seconds for new messages.
    
    Args:
        params: {offset?, limit?, timeout?, allowed_updates?}
        secrets: {TELEGRAM_BOT_TOKEN}
    
    Returns:
        {"ok": bool, "messages": [...], "next_offset": int}
    """
    token = (secrets.get("TELEGRAM_BOT_TOKEN") or 
             params.get("token") or
             "")
    
    if not token:
        return {"error": "TELEGRAM_BOT_TOKEN required (set in vault or pass as token param)"}
    
    # Get parameters for long polling
    offset = params.get("offset")  # int, last update ID + 1 to resume
    limit = params.get("limit", 100)  # max messages to fetch
    timeout = params.get("timeout", 30)  # long poll timeout in seconds
    allowed_updates = params.get("allowed_updates", ["message", "edited_message"])
    
    # Build request
    api_url = f"https://api.telegram.org/bot{token}/getUpdates"
    
    payload = {"limit": limit, "timeout": timeout, "allowed_updates": allowed_updates}
    if offset is not None:
        payload["offset"] = offset
    
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                updates = result.get("result", [])
                messages = []
                next_offset = None
                
                for update in updates:
                    if "message" in update:
                        msg = update["message"]
                        messages.append({
                            "update_id": update.get("update_id"),
                            "message_id": msg.get("message_id"),
                            "chat_id": msg.get("chat", {}).get("id"),
                            "chat_title": msg.get("chat", {}).get("title"),
                            "chat_username": msg.get("chat", {}).get("username"),
                            "from_id": msg.get("from", {}).get("id"),
                            "from_username": msg.get("from", {}).get("username"),
                            "text": msg.get("text"),
                            "date": msg.get("date"),
                        })
                    elif "edited_message" in update:
                        msg = update["edited_message"]
                        messages.append({
                            "update_id": update.get("update_id"),
                            "message_id": msg.get("message_id"),
                            "chat_id": msg.get("chat", {}).get("id"),
                            "chat_username": msg.get("chat", {}).get("username"),
                            "from_id": msg.get("from", {}).get("id"),
                            "from_username": msg.get("from", {}).get("username"),
                            "text": msg.get("text"),
                            "date": msg.get("date"),
                            "type": "edited",
                        })
                    
                    # Track highest update_id for next offset
                    if next_offset is None or update.get("update_id", 0) > next_offset:
                        next_offset = update.get("update_id", 0)
                
                # Next offset is last update_id + 1 (Telegram requires this)
                if next_offset is not None:
                    next_offset += 1
                
                return {
                    "ok": True,
                    "messages": messages,
                    "count": len(messages),
                    "next_offset": next_offset,
                }
            else:
                return {
                    "ok": False,
                    "error_code": result.get("error_code"),
                    "description": result.get("description"),
                }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            return {"error": f"HTTP {e.code}", "details": error_body}
        except Exception:
            return {"error": f"HTTP {e.code}", "body": e.read().decode()[:512]}
    except Exception as e:
        return {"error": str(e)}


def handle_set_webhook(params: dict, secrets: dict) -> dict:
    """"Set a webhook for incoming Telegram updates.
    
    Args:
        params: {url, allowed_updates?, secret_token?}
        secrets: {TELEGRAM_BOT_TOKEN}
    
    Returns:
        {"ok": bool, "result": {...}}
    """
    token = (secrets.get("TELEGRAM_BOT_TOKEN") or 
             params.get("token") or
             "")
    url = params.get("url", "")
    
    if not token:
        return {"error": "TELEGRAM_BOT_TOKEN required (set in vault or pass as token param)"}
    if not url:
        return {"error": "url param required"}
    
    api_url = f"https://api.telegram.org/bot{token}/setWebhook"
    
    payload = {"url": url}
    if params.get("allowed_updates"):
        payload["allowed_updates"] = params["allowed_updates"]
    if params.get("secret_token"):
        payload["secret_token"] = params["secret_token"]
    
    data = json.dumps(payload).encode()
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result
    except Exception as e:
        return {"error": str(e)}


def handle_delete_webhook(params: dict, secrets: dict) -> dict:
    """Delete the webhook.
    
    Args:
        params: {}  (no params needed)
        secrets: {TELEGRAM_BOT_TOKEN}
    
    Returns:
        {"ok": bool, "result": bool}
    """
    token = (secrets.get("TELEGRAM_BOT_TOKEN") or 
             params.get("token") or
             "")
    
    if not token:
        return {"error": "TELEGRAM_BOT_TOKEN required (set in vault or pass as token param)"}
    
    api_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    
    req = urllib.request.Request(api_url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

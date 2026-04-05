"""
Slack API skill for AgentOS.

Send messages and interact with Slack via Web API.
https://api.slack.com/methods

Secrets required (via vault or env):
- SLACK_TOKEN: Bot user OAuth token (xoxb-...)
- SLACK_DEFAULT_CHANNEL: Default channel ID to post to (optional)

Params:
- channel: Override default channel (name or ID)
- text: Message text to send
- blocks: JSON array of Slack Block Kit elements
- attachments: JSON array of legacy attachments

Additional operations:
- slack.conversations_list: List channels
- slack.users_info: Get user info
- slack.webhook: Send via incoming webhook (simpler, no token needed)
"""

import json
import urllib.request
import urllib.error

SKILL_ID = "slack.send"


def handle(params: dict, secrets: dict) -> dict:
    """
    Send a message via Slack API chat.postMessage.
    
    Args:
        params: {channel?, text, blocks?, attachments?}
        secrets: {SLACK_TOKEN, SLACK_DEFAULT_CHANNEL?}
    
    Returns:
        {"ok": bool, "ts": str, "channel": str, "message": dict}
    """
    # Get secrets - check vault first, then env vars
    token = (secrets.get("SLACK_TOKEN") or 
             params.get("token") or
             "")
    default_channel = secrets.get("SLACK_DEFAULT_CHANNEL", "")

    # Get message params
    channel = (params.get("channel") or 
               default_channel or
               "")
    text = params.get("text", "")
    blocks = params.get("blocks", "")
    attachments = params.get("attachments", "")

    # Validation
    if not token:
        return {"error": "SLACK_TOKEN required (set in vault or pass as token param)"}
    if not channel:
        return {"error": "channel param or SLACK_DEFAULT_CHANNEL required"}
    if not text:
        return {"error": "text param required"}

    # Convert channel name to ID if it starts with #
    if channel.startswith("#"):
        channel = channel[1:]  # Slack API expects ID, not name - but we try anyway
    
    # Build API URL
    api_url = "https://slack.com/api/chat.postMessage"
    
    # Build payload
    payload = {
        "channel": channel,
        "text": text,
    }
    
    # Add optional elements
    if blocks:
        try:
            if isinstance(blocks, str):
                blocks = json.loads(blocks)
            payload["blocks"] = blocks
        except json.JSONDecodeError:
            return {"error": "blocks must be valid JSON array"}
    
    if attachments:
        try:
            if isinstance(attachments, str):
                attachments = json.loads(attachments)
            payload["attachments"] = attachments
        except json.JSONDecodeError:
            return {"error": "attachments must be valid JSON array"}
    
    # Set username and icon if provided
    username = params.get("username")
    if username:
        payload["username"] = username
    
    icon_url = params.get("icon_url")
    if icon_url:
        payload["icon_url"] = icon_url
    
    icon_emoji = params.get("icon_emoji")
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji

    # Make request
    data = urllib.parse.urlencode({"payload": json.dumps(payload)}).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                return {
                    "ok": True,
                    "ts": result.get("ts"),
                    "channel": result.get("channel"),
                    "message": result.get("message", {}),
                }
            else:
                return {
                    "ok": False,
                    "error": result.get("error", "Unknown error"),
                    "detail": result,
                }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            return {"ok": False, "error": f"HTTP {e.code}", "details": error_body}
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}", "body": e.read().decode()[:512]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_conversations_list(params: dict, secrets: dict) -> dict:
    """
    List Slack channels/conversations via conversations.list API.
    
    Args:
        params: {types?, limit?, cursor?}
        secrets: {SLACK_TOKEN}
    
    Returns:
        {"ok": bool, "channels": [...], "response_metadata": dict}
    """
    token = secrets.get("SLACK_TOKEN") or params.get("token")
    
    if not token:
        return {"error": "SLACK_TOKEN required"}
    
    api_url = "https://slack.com/api/conversations.list"
    
    # Build payload
    payload = {
        "types": params.get("types", "public_channel,private_channel"),
        "limit": params.get("limit", 100),
    }
    
    cursor = params.get("cursor")
    if cursor:
        payload["cursor"] = cursor
    
    data = urllib.parse.urlencode(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                return {
                    "ok": True,
                    "channels": result.get("channels", []),
                    "response_metadata": result.get("response_metadata", {}),
                }
            return {"ok": False, "error": result.get("error", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_users_info(params: dict, secrets: dict) -> dict:
    """
    Get Slack user info via users.info API.
    
    Args:
        params: {user} - User ID (U...) or username
        secrets: {SLACK_TOKEN}
    
    Returns:
        {"ok": bool, "user": dict}
    """
    token = secrets.get("SLACK_TOKEN") or params.get("token")
    user = params.get("user", "")
    
    if not token:
        return {"error": "SLACK_TOKEN required"}
    if not user:
        return {"error": "user param required (user ID or username)"}
    
    api_url = "https://slack.com/api/users.info"
    
    payload = {"user": user}
    data = urllib.parse.urlencode(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                return {
                    "ok": True,
                    "user": result.get("user", {}),
                }
            return {"ok": False, "error": result.get("error", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_webhook(params: dict, secrets: dict) -> dict:
    """
    Send a message via Slack Incoming Webhook.
    Simpler than full API - no OAuth token needed.
    
    Args:
        params: {webhook_url, text, blocks?, attachments?}
        secrets: {} - webhook URL provided in params
    
    Returns:
        {"ok": bool}
    """
    webhook_url = params.get("webhook_url", "")
    text = params.get("text", "")
    blocks = params.get("blocks", "")
    attachments = params.get("attachments", "")
    
    if not webhook_url:
        return {"error": "webhook_url param required"}
    if not text:
        return {"error": "text param required"}
    
    # Build payload
    payload = {"text": text}
    
    if blocks:
        try:
            if isinstance(blocks, str):
                blocks = json.loads(blocks)
            payload["blocks"] = blocks
        except json.JSONDecodeError:
            pass
    
    if attachments:
        try:
            if isinstance(attachments, str):
                attachments = json.loads(attachments)
            payload["attachments"] = attachments
        except json.JSONDecodeError:
            pass
    
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    
    req = urllib.request.Request(webhook_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Webhooks typically return empty body on success
            return {"ok": resp.status == 200}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_update(params: dict, secrets: dict) -> dict:
    """
    Update an existing Slack message via chat.update.
    
    Args:
        params: {channel, ts, text, blocks?}
        secrets: {SLACK_TOKEN}
    
    Returns:
        {"ok": bool, "ts": str}
    """
    token = secrets.get("SLACK_TOKEN") or params.get("token")
    channel = params.get("channel", "")
    ts = params.get("ts", "")
    text = params.get("text", "")
    blocks = params.get("blocks", "")
    
    if not token:
        return {"error": "SLACK_TOKEN required"}
    if not channel or not ts or not text:
        return {"error": "channel, ts, and text required"}
    
    api_url = "https://slack.com/api/chat.update"
    
    payload = {
        "channel": channel,
        "ts": ts,
        "text": text,
    }
    
    if blocks:
        try:
            if isinstance(blocks, str):
                blocks = json.loads(blocks)
            payload["blocks"] = blocks
        except json.JSONDecodeError:
            pass
    
    data = urllib.parse.urlencode({"payload": json.dumps(payload)}).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if result.get("ok"):
                return {
                    "ok": True,
                    "ts": result.get("ts"),
                }
            return {"ok": False, "error": result.get("error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_delete(params: dict, secrets: dict) -> dict:
    """
    Delete a Slack message via chat.delete.
    
    Args:
        params: {channel, ts}
        secrets: {SLACK_TOKEN}
    
    Returns:
        {"ok": bool}
    """
    token = secrets.get("SLACK_TOKEN") or params.get("token")
    channel = params.get("channel", "")
    ts = params.get("ts", "")
    
    if not token:
        return {"error": "SLACK_TOKEN required"}
    if not channel or not ts:
        return {"error": "channel and ts required"}
    
    api_url = "https://slack.com/api/chat.delete"
    
    payload = {"channel": channel, "ts": ts}
    data = urllib.parse.urlencode(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return {"ok": result.get("ok", False)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
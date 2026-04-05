"""
WhatsApp Business Cloud API skill for AgentOS.

Send messages via WhatsApp Business Cloud API (Meta).
https://developers.facebook.com/docs/whatsapp/cloud-api

Secrets required (via vault or env):
- WHATSAPP_TOKEN: Access token from Meta Developer Portal
- WHATSAPP_PHONE_ID: Phone number ID from Meta Developer Portal

Params:
- to: Recipient phone number (E.164 format, e.g., +1234567890)
- text: Message text to send
- type: Message type (text, image, document, audio, video, sticker)
- media_url: URL for media attachments (for image, document, audio, video, sticker)
- caption: Caption for media messages
"""

import json
import urllib.request
import urllib.error

SKILL_ID = "whatsapp.send"


def handle(params: dict, secrets: dict) -> dict:
    """
    Send a message via WhatsApp Business Cloud API.
    
    Args:
        params: {to, text?, type?, media_url?, caption?}
        secrets: {WHATSAPP_TOKEN, WHATSAPP_PHONE_ID?}
    
    Returns:
        {"ok": bool, "message_id": str, "to": str}
    """
    # Get secrets - check vault first, then env vars
    token = (secrets.get("WHATSAPP_TOKEN") or 
             params.get("token") or
             "")
    phone_id = (secrets.get("WHATSAPP_PHONE_ID") or 
                params.get("phone_id") or
                "")

    # Get message params
    to = params.get("to", "")
    text = params.get("text", "")
    message_type = params.get("type", "text")
    media_url = params.get("media_url", "")
    caption = params.get("caption", "")

    # Validation
    if not token:
        return {"error": "WHATSAPP_TOKEN required (set in vault or pass as token param)"}
    if not phone_id:
        return {"error": "WHATSAPP_PHONE_ID required (set in vault or pass as phone_id param)"}
    if not to:
        return {"error": "to (phone number) param required in E.164 format (e.g., +1234567890)"}

    # Build API URL
    api_url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    
    # Build message payload based on type
    if message_type == "text":
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text}
        }
    elif message_type in ("image", "document", "audio", "video", "sticker"):
        if not media_url:
            return {"error": f"media_url required for {message_type} type"}
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": message_type,
            message_type: {
                "link": media_url
            }
        }
        if caption:
            payload[message_type]["caption"] = caption
    else:
        return {"error": f"Unknown message type: {message_type}. Use: text, image, document, audio, video, sticker"}

    # Make request
    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if "messages" in result and len(result["messages"]) > 0:
                return {
                    "ok": True,
                    "message_id": result["messages"][0]["id"],
                    "to": to,
                    "type": message_type,
                }
            else:
                return {
                    "ok": False,
                    "error": "No message ID returned",
                    "response": result,
                }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            return {
                "ok": False,
                "error": f"HTTP {e.code}",
                "details": error_body.get("error", {}),
            }
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}", "body": e.read().decode()[:512]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_list(params: dict, secrets: dict) -> dict:
    """
    Send an interactive list message via WhatsApp Business Cloud API.
    
    Args:
        params: {to, title, message, button_text, sections}
        secrets: {WHATSAPP_TOKEN, WHATSAPP_PHONE_ID?}
    
    Returns:
        {"ok": bool, "message_id": str, "to": str}
    """
    token = secrets.get("WHATSAPP_TOKEN") or params.get("token")
    phone_id = secrets.get("WHATSAPP_PHONE_ID") or params.get("phone_id")
    to = params.get("to", "")
    title = params.get("title", "")
    message = params.get("message", "")
    button_text = params.get("button_text", "Select an option")
    sections = params.get("sections", [])
    
    if not token or not phone_id or not to:
        return {"error": "WHATSAPP_TOKEN, WHATSAPP_PHONE_ID, and to (phone) required"}
    
    if not title or not message or not sections:
        return {"error": "title, message, and sections required"}
    
    api_url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "title": title,
            "body": message,
            "footer": params.get("footer", ""),
            "action": {
                "button": button_text,
                "sections": sections
            }
        }
    }
    
    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if "messages" in result and len(result["messages"]) > 0:
                return {
                    "ok": True,
                    "message_id": result["messages"][0]["id"],
                    "to": to,
                    "type": "list",
                }
            return {"ok": False, "response": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_template(params: dict, secrets: dict) -> dict:
    """
    Send a template message via WhatsApp Business Cloud API.
    
    Args:
        params: {to, template_name, language?, components?}
        secrets: {WHATSAPP_TOKEN, WHATSAPP_PHONE_ID?}
    
    Returns:
        {"ok": bool, "message_id": str, "to": str}
    """
    token = secrets.get("WHATSAPP_TOKEN") or params.get("token")
    phone_id = secrets.get("WHATSAPP_PHONE_ID") or params.get("phone_id")
    to = params.get("to", "")
    template_name = params.get("template_name", "")
    language = params.get("language", "en_US")
    components = params.get("components", [])
    
    if not token or not phone_id or not to:
        return {"error": "WHATSAPP_TOKEN, WHATSAPP_PHONE_ID, and to (phone) required"}
    
    if not template_name:
        return {"error": "template_name required"}
    
    api_url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language},
        }
    }
    
    if components:
        payload["template"]["components"] = components
    
    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            if "messages" in result and len(result["messages"]) > 0:
                return {
                    "ok": True,
                    "message_id": result["messages"][0]["id"],
                    "to": to,
                    "type": "template",
                    "template": template_name,
                }
            return {"ok": False, "response": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_mark_seen(params: dict, secrets: dict) -> dict:
    """
    Mark a message as read (send read receipt) via WhatsApp Business Cloud API.
    
    Args:
        params: {to, message_id}
        secrets: {WHATSAPP_TOKEN, WHATSAPP_PHONE_ID?}
    
    Returns:
        {"ok": bool}
    """
    token = secrets.get("WHATSAPP_TOKEN") or params.get("token")
    phone_id = secrets.get("WHATSAPP_PHONE_ID") or params.get("phone_id")
    to = params.get("to", "")
    message_id = params.get("message_id", "")
    
    if not token or not phone_id or not to:
        return {"error": "WHATSAPP_TOKEN, WHATSAPP_PHONE_ID, and to required"}
    
    if not message_id:
        return {"error": "message_id required"}
    
    api_url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "request_read_receipt",
        "request_read_receipt": {
            "message_id": message_id
        }
    }
    
    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return {"ok": True, "response": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
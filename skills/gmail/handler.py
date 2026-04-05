"""
Gmail API skill for AgentOS.

Send and read emails via Gmail API.
https://developers.google.com/gmail/api

Secrets required (via vault or env):
- GMAIL_CLIENT_ID: OAuth2 client ID from Google Cloud Console
- GMAIL_CLIENT_SECRET: OAuth2 client secret
- GMAIL_REFRESH_TOKEN: Refresh token obtained via OAuth2 flow

OAuth2 flow for Pi Zero (must be done on another machine):
1. Go to Google Cloud Console → Credentials → Create OAuth2 client ID
2. Set redirect URI to "http://localhost"
3. Use Google's OAuth2 library or manual flow to get refresh_token
4. Store refresh_token in vault

Params:
- mode: Operation mode: "send", "list", "get", "draft"
- Additional params based on mode (see each function)
"""

import base64
import json
import urllib.request
import urllib.error
import urllib.parse

SKILL_ID = "gmail.send"


def _get_access_token(secrets: dict, params: dict) -> str:
    """
    Exchange refresh token for access token.
    
    Returns access token or raises exception.
    """
    client_id = secrets.get("GMAIL_CLIENT_ID") or params.get("client_id")
    client_secret = secrets.get("GMAIL_CLIENT_SECRET") or params.get("client_secret")
    refresh_token = secrets.get("GMAIL_REFRESH_TOKEN") or params.get("refresh_token")
    
    if not client_id or not client_secret or not refresh_token:
        raise ValueError("GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN required")
    
    # Exchange refresh token for access token
    token_url = "https://oauth2.googleapis.com/token"
    
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    
    req = urllib.request.Request(token_url, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("access_token", "")
    except Exception as e:
        raise RuntimeError(f"Failed to get access token: {e}")


def handle(params: dict, secrets: dict) -> dict:
    """
    Send or manage emails via Gmail API.
    
    Main handler that routes to sub-operations based on 'mode' param.
    
    Args:
        params: {mode, ...}
            mode="send": {to, subject, body, from?, cc?, bcc?, attachments?}
            mode="list": {max_results?, query?, label_ids?}
            mode="get": {message_id, format?}
            mode="draft": {to, subject, body}
        secrets: {GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN}
    
    Returns:
        Depends on mode - see sub-functions
    """
    mode = params.get("mode", "send")
    
    if mode == "send":
        return handle_send(params, secrets)
    elif mode == "list":
        return handle_list(params, secrets)
    elif mode == "get":
        return handle_get(params, secrets)
    elif mode == "draft":
        return handle_draft(params, secrets)
    else:
        return {"error": f"Unknown mode: {mode}. Use: send, list, get, draft"}


def handle_send(params: dict, secrets: dict) -> dict:
    """
    Send an email via Gmail API.
    
    Args:
        params: {to, subject, body, from?, cc?, bcc?, html?}
        secrets: {GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN}
    
    Returns:
        {"ok": bool, "message_id": str}
    """
    try:
        access_token = _get_access_token(secrets, params)
    except ValueError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    
    to = params.get("to", "")
    subject = params.get("subject", "")
    body = params.get("body", "")
    from_addr = params.get("from")
    cc = params.get("cc", "")
    bcc = params.get("bcc", "")
    html = params.get("html", "")
    
    if not to:
        return {"error": "to (recipient email) required"}
    if not subject:
        return {"error": "subject required"}
    if not body and not html:
        return {"error": "body or html required"}
    
    # Build email MIME message
    message = _build_mime_message(
        to=to,
        subject=subject,
        body=body,
        html=html,
        from_addr=from_addr,
        cc=cc,
        bcc=bcc
    )
    
    # Encode to base64url
    encoded_message = base64.urlsafe_b64encode(message.encode()).decode()
    
    # Send via Gmail API
    api_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    
    payload = json.dumps({"raw": encoded_message}).encode()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return {
                "ok": True,
                "message_id": result.get("id"),
                "thread_id": result.get("threadId"),
            }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            return {"ok": False, "error": f"HTTP {e.code}", "details": error_body.get("error", {})}
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_list(params: dict, secrets: dict) -> dict:
    """
    List emails via Gmail API.
    
    Args:
        params: {max_results?, query?, label_ids?}
        secrets: {GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN}
    
    Returns:
        {"ok": bool, "messages": [...], "result_size_estimate": int}
    """
    try:
        access_token = _get_access_token(secrets, params)
    except ValueError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    
    max_results = params.get("max_results", 10)
    query = params.get("query", "")
    label_ids = params.get("label_ids", "")
    
    # Build API URL
    api_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    
    query_params = {"maxResults": max_results}
    if query:
        query_params["q"] = query
    if label_ids:
        if isinstance(label_ids, str):
            label_ids = label_ids.split(",")
        query_params["labelIds"] = label_ids
    
    url = api_url + "?" + urllib.parse.urlencode(query_params)
    
    headers = {"Authorization": f"Bearer {access_token}"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            messages = []
            for msg in result.get("messages", []):
                messages.append({
                    "id": msg.get("id"),
                    "thread_id": msg.get("threadId"),
                })
            
            return {
                "ok": True,
                "messages": messages,
                "result_size_estimate": result.get("resultSizeEstimate", 0),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_get(params: dict, secrets: dict) -> dict:
    """
    Get a specific email via Gmail API.
    
    Args:
        params: {message_id, format?}
            format: "full", "metadata", "raw", "minimal" (default: "full")
        secrets: {GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN}
    
    Returns:
        {"ok": bool, "message": dict}
    """
    try:
        access_token = _get_access_token(secrets, params)
    except ValueError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    
    message_id = params.get("message_id", "")
    fmt = params.get("format", "full")
    
    if not message_id:
        return {"error": "message_id required"}
    
    api_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
    
    query_params = {"format": fmt}
    url = api_url + "?" + urllib.parse.urlencode(query_params)
    
    headers = {"Authorization": f"Bearer {access_token}"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            
            # Parse headers for easier access
            headers_dict = {}
            for h in result.get("payload", {}).get("headers", []):
                headers_dict[h["name"].lower()] = h["value"]
            
            return {
                "ok": True,
                "message_id": result.get("id"),
                "thread_id": result.get("threadId"),
                "subject": headers_dict.get("subject", ""),
                "from": headers_dict.get("from", ""),
                "to": headers_dict.get("to", ""),
                "date": headers_dict.get("date", ""),
                "snippet": result.get("snippet", ""),
                "payload": result.get("payload", {}),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def handle_draft(params: dict, secrets: dict) -> dict:
    """
    Create a draft email via Gmail API.
    
    Args:
        params: {to, subject, body, from?, cc?, bcc?}
        secrets: {GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN}
    
    Returns:
        {"ok": bool, "draft_id": str}
    """
    try:
        access_token = _get_access_token(secrets, params)
    except ValueError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    
    to = params.get("to", "")
    subject = params.get("subject", "")
    body = params.get("body", "")
    from_addr = params.get("from")
    cc = params.get("cc", "")
    bcc = params.get("bcc", "")
    
    if not to:
        return {"error": "to (recipient email) required"}
    if not subject:
        return {"error": "subject required"}
    if not body:
        return {"error": "body required"}
    
    # Build email MIME message
    message = _build_mime_message(
        to=to,
        subject=subject,
        body=body,
        from_addr=from_addr,
        cc=cc,
        bcc=bcc
    )
    
    # Encode to base64url
    encoded_message = base64.urlsafe_b64encode(message.encode()).decode()
    
    # Create draft via Gmail API
    api_url = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
    
    payload = json.dumps({
        "message": {"raw": encoded_message}
    }).encode()
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return {
                "ok": True,
                "draft_id": result.get("id"),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _build_mime_message(to: str, subject: str, body: str = "", html: str = "",
                        from_addr: str = None, cc: str = "", bcc: str = "") -> str:
    """
    Build a raw MIME message string.
    """
    import email.utils
    
    lines = []
    
    # From header (optional - defaults to authenticated user)
    if from_addr:
        lines.append(f"From: {from_addr}")
    
    # To header
    lines.append(f"To: {to}")
    
    # CC header
    if cc:
        lines.append(f"Cc: {cc}")
    
    # BCC header
    if bcc:
        lines.append(f"Bcc: {bcc}")
    
    # Subject
    lines.append(f"Subject: {subject}")
    
    # Date
    lines.append(f"Date: {email.utils.formatdate()}")
    
    # Message-ID
    lines.append(f"Message-ID: {email.utils.make_msgid()}")
    
    # MIME-Version
    lines.append("MIME-Version: 1.0")
    
    # Content-Type
    if html:
        # Multi-part alternative
        boundary = "----=_Part_0_1234567890.9876543210"
        lines.append(f"Content-Type: multipart/alternative; boundary={boundary}")
        lines.append("")
        
        # Plain text part
        lines.append(f"--{boundary}")
        lines.append("Content-Type: text/plain; charset=UTF-8")
        lines.append("Content-Transfer-Encoding: 7bit")
        lines.append("")
        lines.append(body)
        lines.append("")
        
        # HTML part
        lines.append(f"--{boundary}")
        lines.append("Content-Type: text/html; charset=UTF-8")
        lines.append("Content-Transfer-Encoding: 7bit")
        lines.append("")
        lines.append(html)
        lines.append("")
        
        # End boundary
        lines.append(f"--{boundary}--")
    else:
        # Plain text only
        lines.append("Content-Type: text/plain; charset=UTF-8")
        lines.append("Content-Transfer-Encoding: 7bit")
        lines.append("")
        lines.append(body)
    
    return "\n".join(lines)
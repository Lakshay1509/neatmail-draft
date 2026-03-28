"""
NeatMail Context Engine
-----------------------
FastAPI/Uvicorn service that retrieves semantic email context from Pinecone
(relationship, topic, behavioural) vs. an incoming email body.

Flow:
  1. Receive  user_id, sender_domain, token (Google OAuth), body
  2. Query Pinecone (metadata filter: user_id + sender_domain) for vector count
  3. If cold → fetch last 3 months of emails from that sender via Gmail search,
     embed + upsert all of them
  4. Embed current body, run filtered similarity search → three context scores
  5. Upsert current body
  6. Return Relationship / Topic / Behavioural context + relevance signal

Namespace strategy
------------------
  Single namespace = user_id   (one per user, NOT one per user+domain)
  Sender domain is a metadata field → filtered via Pinecone metadata filter.
  This keeps namespace count = number of users, not users × senders.
"""

import asyncio
import base64
from email.utils import parsedate_to_datetime
import html
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("context-engine")

# ---------------------------------------------------------------------------
# Env / Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY      = os.environ["AZURE_API_KEY"]
OPENAI_ENDPOINT     = os.environ["AZURE_ENDPOINT"]
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neatmail-context")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL         = "text-embedding-3-small"
EMBED_DIM           = 1536            # text-embedding-3-small native dimension

# Minimum stored vectors for this user+sender before skipping history fetch
MIN_VECTORS         = int(os.getenv("MIN_VECTORS", "5"))

# How many neighbours to pull for each context query
TOP_K               = int(os.getenv("TOP_K", "10"))

# Relevance threshold (cosine similarity)
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.75"))

# Maximum emails to pull from 2-month history (keeps embedding costs sane)
HISTORY_MAX_EMAILS  = int(os.getenv("HISTORY_MAX_EMAILS", "50"))

# Look-back window in days
HISTORY_DAYS        = int(os.getenv("HISTORY_DAYS", "60"))

GMAIL_API_BASE      = "https://gmail.googleapis.com/gmail/v1/users/me"

# ---------------------------------------------------------------------------
# Clients (module-level singletons)
# ---------------------------------------------------------------------------
openai_client = OpenAI(
        base_url=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY
    )

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist yet
existing = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing:
    log.info("Creating Pinecone index '%s' …", PINECONE_INDEX_NAME)
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
    log.info("Index ready.")

index = pc.Index(PINECONE_INDEX_NAME)

# ---------------------------------------------------------------------------
# Warm-pair cache  (in-process; avoids an extra Pinecone RU per request)
# ---------------------------------------------------------------------------
# Stores "user_id::sender_domain" strings for pairs that have already been
# initialised this process lifetime.  On restart it's empty, so the first
# request after restart re-checks via a real query (one-time cost per pair).
_warm_pairs: set[str] = set()    #memory usage dekhni padegi
_warm_lock = asyncio.Lock()   # prevents duplicate cold-start on burst traffic

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NeatMail Context Engine",
    description="Semantic email relationship/topic/behavioural context via Pinecone + OpenAI",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ContextRequest(BaseModel):
    user_id:       str = Field(..., description="Clerk / internal user ID")
    sender_email:  str = Field(..., description="Full sender email address, e.g. 'john@github.com'")
    token:         str = Field(..., description="Mailbox OAuth2 access token (Gmail or Microsoft Graph)")
    body:          str = Field(..., description="Current email body to analyse")
    subject:       Optional[str] = Field(None, description="Email subject (optional, improves quality)")
    timezone:      str = Field(..., description="User timezone, e.g. 'America/New_York'")
    is_gmail:      bool = Field(True, description="Whether the mailbox token is for Gmail. Defaults to true.")

class ContextScore(BaseModel):
    description: str

class MentionedDate(BaseModel):
    raw: str
    iso: str

class ContextResponse(BaseModel):
    relationship_context: ContextScore
    topic_context:        ContextScore
    behavioural_context:  ContextScore
    overall_relevance:    float
    is_relevant:          bool
    vectors_upserted:     int
    user_namespace:       str
    sender_email:         str
    keywords:             list[str] = Field(default_factory=list)
    mentionedDates:       list[MentionedDate] = Field(default_factory=list)
    intent:               str = Field(default="general")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _user_namespace(user_id: str) -> str:
    """
    One Pinecone namespace per USER (not per user+domain).
    sender_domain is stored in metadata and used as a filter on every query.
    This prevents namespace explosion on the free tier.
    """
    return user_id


def _vector_id(text: str, prefix: str = "") -> str:
    """Deterministic SHA-1 based vector ID."""
    digest = hashlib.sha1(text.encode()).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Automatically chunks into batches of 512 (OpenAI limit is 2048 but
    keeping small to avoid timeout on free-tier).
    """
    if not texts:
        return []
    all_vectors: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[i:i + batch_size],
        )
        all_vectors.extend(item.embedding for item in resp.data)
    return all_vectors


def _is_warm(user_id: str, sender_email: str) -> bool:
    """Return True if this user+sender pair is already in the warm cache."""
    return f"{user_id}::{sender_email}" in _warm_pairs


def _mark_warm(user_id: str, sender_email: str) -> None:
    """Mark a user+sender pair as warm so we skip cold-start next time."""
    _warm_pairs.add(f"{user_id}::{sender_email}")

# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

async def _fetch_sender_history(
    sender_email: str,
    token: str,
    days: int = HISTORY_DAYS,
    max_emails: int = HISTORY_MAX_EMAILS,
) -> list[dict]:
    """
    Fetch up to `max_emails` messages from the last `days` days sent by or sent to
    `sender_email` using Gmail search.
    
    Gmail query: (from:{sender_email} OR to:{sender_email}) after:{after_ts}
    """
    headers  = {"Authorization": f"Bearer {token}"}
    cutoff   = datetime.now(timezone.utc) - timedelta(days=days)
    # Gmail's after: filter uses Unix epoch (seconds)
    after_ts = int(cutoff.timestamp())
    query    = f"(from:{sender_email} OR to:{sender_email}) after:{after_ts}"

    message_ids: list[str] = []
    page_token: Optional[str] = None

    log.info("Fetching Gmail history | query='%s' max=%d", query, max_emails)

    async with httpx.AsyncClient(timeout=30) as client:
        # --- Page through message list (lightweight, only IDs) ---
        while len(message_ids) < max_emails:
            params: dict = {
                "q":          query,
                "maxResults": min(50, max_emails - len(message_ids)),
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(
                f"{GMAIL_API_BASE}/messages",
                headers=headers,
                params=params,
            )
            if resp.status_code != 200:
                log.warning("Gmail list failed: %s %s", resp.status_code, resp.text[:200])
                break

            data = resp.json()
            for m in data.get("messages", []):
                message_ids.append(m["id"])

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        log.info("Found %d message IDs for email '%s'", len(message_ids), sender_email)

        # --- Fetch each message in full  (truly parallel via asyncio.gather) ---
        parsed: list[dict] = []

        async def _fetch_one(mid: str) -> Optional[dict]:
            try:
                r = await client.get(
                    f"{GMAIL_API_BASE}/messages/{mid}",
                    headers=headers,
                    params={"format": "full"},
                )
                if r.status_code != 200:
                    return None
                msg     = r.json()
                payload = msg.get("payload", {})
                hdrs    = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
                body    = _extract_body(payload)
                if not body.strip():
                    return None
                return {
                    "id":      msg["id"],
                    "subject": hdrs.get("subject", ""),
                    "from":    hdrs.get("from", ""),
                    "date":    hdrs.get("date", ""),
                    "body":    body,
                }
            except Exception as e:
                log.warning("Error fetching message %s: %s", mid, e)
                return None

        # Batches of 10 in parallel (respects Gmail API rate limits)
        for batch_start in range(0, len(message_ids), 10):
            batch   = message_ids[batch_start:batch_start + 10]
            results = await asyncio.gather(*[_fetch_one(mid) for mid in batch])
            parsed.extend(r for r in results if r is not None)

    log.info("Parsed %d messages with body for email '%s'", len(parsed), sender_email)
    return parsed


def _strip_html(value: str) -> str:
    """Best-effort HTML to text conversion for Outlook bodies."""
    if not value:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


async def _fetch_outlook_messages(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    params: dict,
    max_emails: int,
    date_field: str,
) -> list[dict]:
    """Fetch paginated Outlook messages from Microsoft Graph."""
    collected: list[dict] = []
    next_url: Optional[str] = url
    next_params: Optional[dict] = params

    while next_url and len(collected) < max_emails:
        resp = await client.get(next_url, headers=headers, params=next_params)
        if resp.status_code != 200:
            log.warning("Microsoft Graph list failed: %s %s", resp.status_code, resp.text[:200])
            break

        data = resp.json()
        for msg in data.get("value", []):
            body_text = _strip_html(msg.get("body", {}).get("content", "")) or msg.get("bodyPreview", "")
            if not body_text.strip():
                continue

            sender = (
                msg.get("from", {})
                .get("emailAddress", {})
                .get("address", "")
            ) or (
                msg.get("sender", {})
                .get("emailAddress", {})
                .get("address", "")
            )

            collected.append({
                "id": msg.get("id", ""),
                "subject": msg.get("subject", ""),
                "from": sender,
                "date": msg.get(date_field, ""),
                "body": body_text,
            })

            if len(collected) >= max_emails:
                break

        next_url = data.get("@odata.nextLink")
        next_params = None

    return collected


async def _fetch_sender_history_outlook(
    sender_email: str,
    token: str,
    days: int = HISTORY_DAYS,
    max_emails: int = HISTORY_MAX_EMAILS,
) -> list[dict]:
    """
    Fetch up to `max_emails` Outlook messages from the last `days` days involving
    `sender_email` using Microsoft Graph.

    We fetch both inbound mail and sent mail so behaviour matches the Gmail path.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Prefer": 'outlook.body-content-type="text"',
    }

    inbound_params = {
        "$top": min(25, max_emails),
        "$select": "id,subject,from,sender,body,bodyPreview,receivedDateTime",
        "$filter": (
            f"receivedDateTime ge {cutoff_iso} and "
            f"(from/emailAddress/address eq '{sender_email}' or sender/emailAddress/address eq '{sender_email}')"
        ),
    }

    outbound_params = {
        "$top": min(25, max_emails),
        "$select": "id,subject,from,sender,body,bodyPreview,sentDateTime,toRecipients,ccRecipients,bccRecipients",
        "$filter": (
            f"sentDateTime ge {cutoff_iso} and ("
            f"toRecipients/any(r:r/emailAddress/address eq '{sender_email}') or "
            f"ccRecipients/any(r:r/emailAddress/address eq '{sender_email}') or "
            f"bccRecipients/any(r:r/emailAddress/address eq '{sender_email}')"
            f")"
        ),
    }

    log.info("Fetching Outlook history | sender='%s' max=%d", sender_email, max_emails)

    async with httpx.AsyncClient(timeout=30) as client:
        inbound, outbound = await asyncio.gather(
            _fetch_outlook_messages(
                client=client,
                url="https://graph.microsoft.com/v1.0/me/messages",
                headers=headers,
                params=inbound_params,
                max_emails=max_emails,
                date_field="receivedDateTime",
            ),
            _fetch_outlook_messages(
                client=client,
                url="https://graph.microsoft.com/v1.0/me/mailFolders/SentItems/messages",
                headers=headers,
                params=outbound_params,
                max_emails=max_emails,
                date_field="sentDateTime",
            ),
        )

    combined: dict[str, dict] = {}
    for msg in inbound + outbound:
        if msg.get("id"):
            combined[msg["id"]] = msg

    parsed = sorted(
        combined.values(),
        key=lambda m: m.get("date", ""),
        reverse=True,
    )[:max_emails]

    log.info("Parsed %d Outlook messages with body for email '%s'", len(parsed), sender_email)
    return parsed


async def _fetch_sender_history_for_provider(
    *,
    sender_email: str,
    token: str,
    is_gmail: bool,
    days: int = HISTORY_DAYS,
    max_emails: int = HISTORY_MAX_EMAILS,
) -> list[dict]:
    """Fetch sender history from Gmail or Outlook based on the request."""
    if is_gmail:
        return await _fetch_sender_history(
            sender_email=sender_email,
            token=token,
            days=days,
            max_emails=max_emails,
        )

    return await _fetch_sender_history_outlook(
        sender_email=sender_email,
        token=token,
        days=days,
        max_emails=max_emails,
    )


def _extract_body(payload: dict, depth: int = 0) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    if depth > 5:
        return ""

    mime      = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime == "text/plain" and body_data:
        try:
            return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="ignore")
        except Exception:
            return ""

    for part in payload.get("parts", []):
        text = _extract_body(part, depth + 1)
        if text:
            return text

    return ""


def _parse_message_datetime(value: str) -> Optional[datetime]:
    """Parse various stored message date formats into UTC datetime."""
    if not value:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # Epoch timestamp (seconds)
    if raw.isdigit():
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except Exception:
            pass

    # ISO datetime (Graph) with optional trailing Z
    iso_candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass

    # RFC2822-style datetime (Gmail Date header)
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _recency_weight(message_dt: Optional[datetime], now_utc: datetime) -> float:
    """
    Explicit TTL decay buckets for historical context.
      0-3 days   -> 1.0
      4-6 days   -> 0.8
      7-15 days  -> 0.6
      16-30 days -> 0.4
      31-60 days -> 0.25
      60+ days   -> 0.1
    """
    if message_dt is None:
        return 0.25

    age_days = max((now_utc - message_dt).total_seconds() / 86400.0, 0.0)

    if age_days <= 3:
        return 1.0
    if age_days <= 6:
        return 0.8
    if age_days <= 15:
        return 0.6
    if age_days <= 30:
        return 0.4
    if age_days <= 60:
        return 0.25
    return 0.1


def _apply_recency_decay(matches: list[dict]) -> list[dict]:
    """Apply explicit recency decay to Pinecone match scores."""
    if not matches:
        return []

    now_utc = datetime.now(timezone.utc)
    weighted_matches: list[dict] = []

    for match in matches:
        cloned = dict(match)
        metadata = cloned.get("metadata") or {}
        msg_dt = _parse_message_datetime(str(metadata.get("date", "")))
        weight = _recency_weight(msg_dt, now_utc)
        raw_score = float(cloned.get("score", 0.0))

        cloned["raw_score"] = raw_score
        cloned["time_weight"] = weight
        cloned["score"] = raw_score * weight
        weighted_matches.append(cloned)

    return weighted_matches

# ---------------------------------------------------------------------------
# Pinecone upsert helpers
# ---------------------------------------------------------------------------

def _build_upsert_records(
    messages:      list[dict],
    namespace:     str,
    sender_email:  str,
    user_id:       str,
) -> int:
    """
    Embed and upsert a list of message dicts into Pinecone.
    sender_email and user_id stored in metadata for filtered querying.
    Returns number of vectors upserted.
    """
    if not messages:
        return 0

    valid = [m for m in messages if m.get("body", "").strip()]
    if not valid:
        return 0

    texts = [
        f"Subject: {m.get('subject','')}\nFrom: {m.get('from','')}\n\n{m.get('body','')}"
        .strip()[:8000]
        for m in valid
    ]

    vectors = _embed(texts)

    records = []
    for msg, vec in zip(valid, vectors):
        records.append({
            "id": _vector_id(msg["id"], prefix="msg-"),
            "values": vec,
            "metadata": {
                "user_id":       user_id,
                "sender_email":  sender_email,
                "message_id":    msg.get("id", ""),
                "subject":       msg.get("subject", "")[:256],
                "sender":        msg.get("from", "")[:256],
                "date":          msg.get("date", "")[:64],
                "body_snippet":  msg.get("body", "")[:512],
                "type":          _classify_type(msg.get("subject", ""), msg.get("body", "")),
            },
        })

    # Batch upsert (Pinecone hard limit: 100 vectors per call)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i:i + batch_size], namespace=namespace)

    log.info(
        "Upserted %d vectors (user=%s email=%s) into namespace '%s'",
        len(records), user_id, sender_email, namespace,
    )
    return len(records)


def _classify_type(subject: str, body: str) -> str:
    """Lightweight heuristic to tag a message type for metadata filtering."""
    text = (subject + " " + body).lower()
    if any(w in text for w in ["unsubscribe", "promotional", "discount", "offer", "deal", "coupon"]):
        return "marketing"
    if any(w in text for w in ["invoice", "receipt", "payment", "order", "confirmation", "shipping"]):
        return "transactional"
    if any(w in text for w in ["re:", "reply", "thanks", "hi ", "hello", "dear", "please", "following up"]):
        return "conversational"
    return "informational"

# ---------------------------------------------------------------------------
# Context scoring
# ---------------------------------------------------------------------------

import json

def _generate_llm_context(body_text: str, subject: str, matches: list[dict], global_matches: list[dict], timezone_str: str) -> dict:
    """Uses gpt-5-mini to generate drafter-focused context and extract entities for the user."""

    utc_now = datetime.now(timezone.utc)
    utc_today_str = utc_now.strftime("%Y-%m-%d")

    # History from THIS specific sender
    sender_history = "\n---\n".join([
        f"Snippet: {m.get('metadata', {}).get('body_snippet', '')[:200]}"
        for m in (matches[:3] if matches else [])
    ])

    # History across ALL senders (to catch cross-conversations, e.g. talking to a manager)
    global_history = "\n---\n".join([
        f"Sender: {m.get('metadata', {}).get('sender', 'Unknown')}\nSnippet: {m.get('metadata', {}).get('body_snippet', '')[:200]}"
        for m in (global_matches[:5] if global_matches else [])
    ])

    system_msg = (
        "You are an email context analyst. Output only valid JSON. "
        "STRICT RULES — violating any is an error:\n"
        "1. Only use information explicitly present in the provided email body or retrieved history snippets. "
        "Never infer, assume, or add facts from outside the given text.\n"
        "2. If a history section is empty ('None found'), use the exact fallback string specified for that field. "
        "Do NOT substitute with advice or guesses.\n"
        "3. For mentionedDates: extract only dates written in the email body; never fabricate. "
        "Resolve relative dates using the current local date in the user's timezone."
    )

    prompt = f"""Timezone: {timezone_str} | UTC date: {utc_today_str}

Subject: {subject}
Email body:
{body_text[:800]}

Sender history (retrieved):
{sender_history if sender_history else "None found."}

Global history (retrieved):
{global_history if global_history else "None found."}

Return JSON with exactly these keys and rules:
- relationship: 1-2 sentences from sender history only. Fallback: "No prior relationship history found."
- topic: 1-2 sentences of facts from global history relevant to this email. Fallback: "No prior topic history found."
- behavioural: 1-2 sentences of reply guidance based only on relationship+topic above. Fallback if both empty: "Insufficient history to provide behavioural guidance."
- intent: one of [scheduling_request, task_assignment, question, follow_up, general] — from email body only.
- keywords: up to 3 keywords from email body. [] if none.
- mentionedDates: [{{"raw": "<exact text>", "iso": "<ISO-8601 with {timezone_str} offset>"}}] for each date in the body. [] if none."""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
           seed=42
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.error("Failed to generate LLM drafter context: %s", e)
        return {
            "relationship": "Error generating relationship context.",
            "topic": "Error generating topic context.",
            "behavioural": "Error generating behavioural context.",
            "intent": "general",
            "keywords": [],
            "mentionedDates": []
        }


def _score_relationship(results: list[dict], desc: str) -> tuple[ContextScore, float]:
    """Relationship score derived from conversational matches."""
    conversational = [r for r in results if r.get("metadata", {}).get("type") == "conversational"]
    avg_score = (
        sum(r["score"] for r in conversational) / len(conversational)
        if conversational else 0.0
    )

    return ContextScore(description=desc), round(avg_score, 4)


def _score_topic(results: list[dict], desc: str) -> tuple[ContextScore, float]:
    """Topic score based on raw top-3 cosine matches."""
    if not results:
        return ContextScore(description=desc), 0.0

    top_scores = sorted([r["score"] for r in results], reverse=True)
    avg_top3   = sum(top_scores[:3]) / min(3, len(top_scores))

    return ContextScore(description=desc), round(avg_top3, 4)


def _score_behavioural(results: list[dict], desc: str) -> tuple[ContextScore, float]:
    """Behavioural score based on the dominant predicted sending pattern."""
    type_counts: dict[str, int]   = {}
    type_scores: dict[str, float] = {}

    for r in results:
        t = r.get("metadata", {}).get("type", "informational")
        type_counts[t] = type_counts.get(t, 0) + 1
        type_scores[t] = type_scores.get(t, 0.0) + r["score"]

    if not type_counts:
        return ContextScore(description=desc), 0.0

    dominant_type  = max(type_counts, key=lambda k: type_counts[k])
    dominant_ratio = type_counts[dominant_type] / len(results)
    dominant_score = type_scores[dominant_type] / type_counts[dominant_type]

    return ContextScore(description=desc), round(dominant_ratio * dominant_score, 4)

# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/context", response_model=ContextResponse, summary="Get email context")
async def get_context(req: ContextRequest):

    
    namespace = _user_namespace(req.user_id)
    log.info(
        "Context request | user=%s email=%s",
        req.user_id, req.sender_email,
    )

    # ── 1. Embed the current body ─────────────────────────────────────────
    body_text = f"Subject: {req.subject or ''}\n\n{req.body}".strip()[:8000]
    [body_embedding] = _embed([body_text])

    # ── 5. Query Pinecone – filtered to this user+sender ──────────────────
    query_result = index.query(
        vector=body_embedding,
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
        filter={
            "user_id":       {"$eq": req.user_id},
            "sender_email":  {"$eq": req.sender_email},
        },
    )
    matches = query_result.get("matches", [])
    matches = _apply_recency_decay(matches)
    vectors_upserted = 0

    # ── 5b. Query Pinecone – global history (all senders) ─────────────────
    global_query_result = index.query(
        vector=body_embedding,
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
        filter={
            "user_id": {"$eq": req.user_id},
        },
    )
    global_matches = global_query_result.get("matches", [])
    global_matches = _apply_recency_decay(global_matches)

    # ── 6. Cold Start Check (uses actual Pinecone vector count via matches) ─
    if not _is_warm(req.user_id, req.sender_email):
        async with _warm_lock:
            if not _is_warm(req.user_id, req.sender_email):
                if len(matches) >= MIN_VECTORS:
                    log.info("Pinecone already warm for (user=%s, email=%s)", req.user_id, req.sender_email)
                    _mark_warm(req.user_id, req.sender_email)
                else:
                    log.info(
                        "Cold start for (user=%s, email=%s) — found only %d prior vectors. "
                        "Fetching last %d days of history …",
                        req.user_id, req.sender_email, len(matches), HISTORY_DAYS,
                    )
                    messages = await _fetch_sender_history_for_provider(
                        sender_email=req.sender_email,
                        token=req.token,
                        is_gmail=req.is_gmail,
                    )

                  

                    if messages:
                        vectors_upserted = _build_upsert_records(
                            messages=messages,
                            namespace=namespace,
                            sender_email=req.sender_email,
                            user_id=req.user_id,
                        )
                        # Re-query Pinecone so the scores include the new history!
                        query_result = index.query(
                            vector=body_embedding,
                            top_k=TOP_K,
                            include_metadata=True,
                            namespace=namespace,
                            filter={
                                "user_id":       {"$eq": req.user_id},
                                "sender_email":  {"$eq": req.sender_email},
                            },
                        )
                        matches = query_result.get("matches", [])
                        matches = _apply_recency_decay(matches)
                        
                        global_query_result = index.query(
                            vector=body_embedding,
                            top_k=TOP_K,
                            include_metadata=True,
                            namespace=namespace,
                            filter={"user_id": {"$eq": req.user_id}},
                        )
                        global_matches = global_query_result.get("matches", [])
                        global_matches = _apply_recency_decay(global_matches)
                    else:
                        log.warning("No history found for email '%s' in the last %d days.", req.sender_email, HISTORY_DAYS)

                    _mark_warm(req.user_id, req.sender_email)

    log.info("Scoring based on %d sender matches / %d global matches for (user=%s)", len(matches), len(global_matches), req.user_id)

    # ── 7. Build the Drafter Context ─────────────────────────────────────
    llm_context = _generate_llm_context(
        body_text=body_text,
        subject=req.subject or "",
        matches=matches,
        global_matches=global_matches,
        timezone_str=req.timezone
    )

    rel_ctx, rel_score = _score_relationship(matches, desc=llm_context.get("relationship", ""))
    top_ctx, top_score = _score_topic(matches, desc=llm_context.get("topic", ""))
    beh_ctx, beh_score = _score_behavioural(matches, desc=llm_context.get("behavioural", ""))

    overall_relevance = round(
        rel_score * 0.30 + top_score * 0.50 + beh_score * 0.20, 4
    )
    is_relevant = overall_relevance >= RELEVANCE_THRESHOLD

    # ── 7. Upsert the current body so future calls benefit from it ────────
    current_record = {
        "id": _vector_id(f"{req.user_id}::{req.sender_email}::{req.body[:100]}", prefix="cur-"),
        "values": body_embedding,
        "metadata": {
            "user_id":       req.user_id,
            "sender_email":  req.sender_email,
            "message_id":    "",
            "subject":       (req.subject or "")[:256],
            "sender":        req.sender_email[:256],
            "date":          str(int(time.time())),
            "body_snippet":  req.body[:512],
            "type":          _classify_type(req.subject or "", req.body),
            "source":        "current",
        },
    }
    index.upsert(vectors=[current_record], namespace=namespace)
    vectors_upserted += 1

    log.info(
        "overall_relevance=%.4f is_relevant=%s (user=%s email=%s)",
        overall_relevance, is_relevant, req.user_id, req.sender_email,
    )

    return ContextResponse(
        relationship_context=rel_ctx,
        topic_context=top_ctx,
        behavioural_context=beh_ctx,
        overall_relevance=overall_relevance,
        is_relevant=is_relevant,
        vectors_upserted=vectors_upserted,
        user_namespace=namespace,
        sender_email=req.sender_email,
        keywords=llm_context.get("keywords", []),
        mentionedDates=llm_context.get("mentionedDates", []),
        intent=llm_context.get("intent", "general"),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": EMBED_MODEL, "index": PINECONE_INDEX_NAME}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")

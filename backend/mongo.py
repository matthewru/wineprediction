# mongo.py
import os, time
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from typing import Optional, List, Dict
from dotenv import load_dotenv, find_dotenv
import certifi

# Load environment variables from .env (searches up the directory tree)
load_dotenv(find_dotenv(), override=False)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "wineapp")

# Optional explicit credentials (avoids URI password encoding issues)
MONGO_USER = os.environ.get("MONGO_USER")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
MONGO_AUTH_SOURCE = os.environ.get("MONGO_AUTH_SOURCE", "admin")

# Configure MongoClient with CA bundle for Atlas connections
_mongo_kwargs = {"serverSelectionTimeoutMS": 8000}
_uri_lower = MONGO_URI.lower()
if _uri_lower.startswith("mongodb+srv://") or "mongodb.net" in _uri_lower or "tls=true" in _uri_lower or "ssl=true" in _uri_lower:
    _mongo_kwargs["tlsCAFile"] = certifi.where()

# Only add username/password if not already present in URI
_has_inline_creds = "@" in MONGO_URI and "://" in MONGO_URI and MONGO_URI.split("://", 1)[1].split("@", 1)[0].count(":") >= 1
if not _has_inline_creds and MONGO_USER and MONGO_PASSWORD:
    _mongo_kwargs["username"] = MONGO_USER
    _mongo_kwargs["password"] = MONGO_PASSWORD
    _mongo_kwargs["authSource"] = MONGO_AUTH_SOURCE

client = MongoClient(MONGO_URI, **_mongo_kwargs)
db = client[MONGO_DB]

def ensure_indexes():
    # users
    db.users.create_index([("handle", ASCENDING)], unique=True, sparse=True)
    db.users.create_index([("google_id", ASCENDING)], unique=True, sparse=True)
    # bottles
    db.bottles.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    db.bottles.create_index([("public", ASCENDING), ("created_at", DESCENDING)])

def now() -> int:
    return int(time.time())

# ---- Initialization ----
def init_mongo() -> bool:
    """Ping the server and ensure indexes. Returns True if connected."""
    try:
        client.admin.command("ping")
        ensure_indexes()
        print("✅ MongoDB connected and indexes ensured")
        return True
    except Exception as e:
        print(f"⚠️  MongoDB initialization failed: {e}")
        return False

# ---- User helpers ----
def get_user_by_id(user_id: str) -> Optional[Dict]:
    return db.users.find_one({"_id": user_id})

def get_user_by_handle(handle: str) -> Optional[Dict]:
    return db.users.find_one({"handle": handle})

def upsert_user(user: Dict) -> str:
    """Create or update a user by _id or handle. Returns the user _id."""
    user_doc = {**user}
    user_id = user_doc.get("_id")
    if not user_id:
        # Fallback to handle as _id if not provided
        handle = user_doc.get("handle")
        if handle:
            user_id = handle
            user_doc["_id"] = user_id
        else:
            # Deterministic fallback id
            user_id = str(now())
            user_doc["_id"] = user_id

    user_doc.setdefault("created_at", now())
    db.users.update_one({"_id": user_id}, {"$set": user_doc}, upsert=True)
    return user_id

# ---- Bottle helpers ----
def add_bottle(user_id: str, bottle: Dict) -> str:
    """Insert a bottle document for a user. Returns inserted _id as str."""
    doc = {**bottle}
    doc["user_id"] = user_id
    doc.setdefault("public", False)
    doc.setdefault("created_at", now())
    result = db.bottles.insert_one(doc)
    return str(result.inserted_id)

def get_bottles_by_user(user_id: str, limit: int = 50) -> List[Dict]:
    cursor = (
        db.bottles.find({"user_id": user_id})
        .sort("created_at", DESCENDING)
        .limit(int(limit))
    )
    return list(cursor)

def get_public_bottles(limit: int = 50) -> List[Dict]:
    cursor = (
        db.bottles.find({"public": True})
        .sort("created_at", DESCENDING)
        .limit(int(limit))
    )
    return list(cursor)

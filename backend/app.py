from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_price_lite import predict_price_lite
from services.predict_rating_lite import predict_rating_lite
from services.predict_flavor import predict_flavor_tags_from_dict, load_model_eagerly, check_models_exist
from services.predict_mouthfeel import predict_mouthfeel_tags_from_dict
from mongo import init_mongo, db
import os

# Added imports for Google auth and JWT
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import jwt
from pymongo import ReturnDocument
from mongo import get_bottles_by_user
from mongo import add_bottle
from mongo import get_public_bottles
from mongo import now
from embedding_utils import build_embedding, DEFAULT_WEIGHTS
import numpy as np
import json
from bson import ObjectId
from typing import List, Tuple

app = Flask(__name__)
CORS(app)

# Initialize MongoDB
print("🗄️ Initializing MongoDB...")
MONGO_READY = init_mongo()

# JWT secret
JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")

# Allowed Google audiences (comma-separated client IDs)
GOOGLE_CLIENT_IDS = [s.strip() for s in os.environ.get("GOOGLE_CLIENT_IDS", "").split(",") if s.strip()]

# ---- Catalog embeddings (for real-wine match) ----
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, 'data')
_VOCAB_PATH = os.path.join(_DATA_DIR, 'global_vocab_v2_normalized.json')
_EMB_PATH = os.path.join(_DATA_DIR, 'catalog_embeddings.npz')
_META_PATH = os.path.join(_DATA_DIR, 'catalog_meta.jsonl')

CATALOG_X: np.ndarray | None = None  # (N, D) float16 L2-normalized
CATALOG_DIM: int | None = None
GLOBAL_VOCAB: dict | None = None
CATALOG_META: list[dict] | None = None
_OFFSETS_COMPUTED: bool = False
_OFFSETS: dict[str, tuple[int, int]] | None = None  # name -> (start, length)

_DEF_EMPTY_LIST: list[str] = []

def _load_vocab_and_catalog():
    global CATALOG_X, CATALOG_DIM, GLOBAL_VOCAB
    global CATALOG_META
    if GLOBAL_VOCAB is None:
        try:
            with open(_VOCAB_PATH, 'r', encoding='utf-8') as f:
                GLOBAL_VOCAB = json.load(f)
            print(f"📚 Loaded vocab from {_VOCAB_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load vocab: {e}")
            GLOBAL_VOCAB = {}
    if CATALOG_X is None:
        try:
            data = np.load(_EMB_PATH)
            CATALOG_X = data['X']  # float16, L2-normalized
            CATALOG_DIM = int(CATALOG_X.shape[1])
            print(f"📦 Loaded catalog embeddings: {CATALOG_X.shape} from {_EMB_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load catalog embeddings: {e}")
    if CATALOG_META is None:
        try:
            meta: list[dict] = []
            with open(_META_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        meta.append(json.loads(line))
                    except Exception:
                        meta.append({})
            CATALOG_META = meta
            print(f"🧾 Loaded catalog meta rows: {len(CATALOG_META)} from {_META_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load catalog meta: {e}")
    # Compute offsets when vocab and catalog are available
    _compute_offsets_if_ready()

def _compute_offsets_if_ready():
    global _OFFSETS_COMPUTED, _OFFSETS
    if _OFFSETS_COMPUTED:
        return
    if GLOBAL_VOCAB is None:
        return
    try:
        flavors_vocab = GLOBAL_VOCAB.get("flavors") or GLOBAL_VOCAB.get("flavor_vocab") or []
        mouthfeel_vocab = GLOBAL_VOCAB.get("mouthfeel") or GLOBAL_VOCAB.get("mouthfeel_vocab") or []
        # Geography
        geo = GLOBAL_VOCAB.get("geography_vocab") or {}
        countries_vocab = GLOBAL_VOCAB.get("countries") or geo.get("countries") or []
        regions1_vocab = GLOBAL_VOCAB.get("regions1") or geo.get("primary_regions") or []
        regions2_vocab = GLOBAL_VOCAB.get("regions2") or geo.get("secondary_regions") or []
        varieties_vocab = GLOBAL_VOCAB.get("varieties") or GLOBAL_VOCAB.get("variety_vocab") or []
        price_bins = GLOBAL_VOCAB.get("price_bins") or GLOBAL_VOCAB.get("price_buckets") or []
        rating_bins = GLOBAL_VOCAB.get("rating_buckets") or []
        age_bins = GLOBAL_VOCAB.get("age_buckets") or []

        parts = [
            ("flavors", len(flavors_vocab)),
            ("mouthfeel", len(mouthfeel_vocab)),
            ("variety", len(varieties_vocab)),
            ("country", len(countries_vocab)),
            ("region1", len(regions1_vocab)),
            ("region2", len(regions2_vocab)),
            ("age", len(age_bins)),
            ("rating", len(rating_bins)),
            ("price", len(price_bins)),
        ]
        start = 0
        offsets: dict[str, tuple[int, int]] = {}
        for name, ln in parts:
            offsets[name] = (start, ln)
            start += ln
        _OFFSETS = offsets
        _OFFSETS_COMPUTED = True
        # Optional sanity check with CATALOG_DIM
        if CATALOG_X is not None and start != CATALOG_X.shape[1]:
            print(f"⚠️  Offset dim mismatch: computed {start}, catalog {CATALOG_X.shape[1]}")
    except Exception as e:
        print(f"⚠️  Failed to compute offsets: {e}")

def _decode_flavor_mouthfeel_from_row(row: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Decode top flavor and mouthfeel tags from a normalized embedding row using block offsets.
    Returns (flavors, mouthfeel) as lists of {tag/confidence} dicts.
    """
    _compute_offsets_if_ready()
    if _OFFSETS is None or GLOBAL_VOCAB is None:
        return [], []
    try:
        flavors_vocab = GLOBAL_VOCAB.get("flavors") or GLOBAL_VOCAB.get("flavor_vocab") or []
        mouthfeel_vocab = GLOBAL_VOCAB.get("mouthfeel") or GLOBAL_VOCAB.get("mouthfeel_vocab") or []
        start_f, len_f = _OFFSETS.get("flavors", (0, 0))
        start_m, len_m = _OFFSETS.get("mouthfeel", (0, 0))
        if len_f <= 0 and len_m <= 0:
            return [], []
        w_f = float(DEFAULT_WEIGHTS.get("flavors", 1.0))
        w_m = float(DEFAULT_WEIGHTS.get("mouthfeel", 0.7))
        # Extract slices and roughly invert weights
        fv = row[start_f:start_f + len_f].astype(np.float32)
        mv = row[start_m:start_m + len_m].astype(np.float32)
        if len_f > 0 and w_f > 0:
            fv = fv / w_f
        if len_m > 0 and w_m > 0:
            mv = mv / w_m
        # Normalize within-slice for pseudo-confidence
        def top_k_from_slice(vec: np.ndarray, vocab: list[str], is_flavor: bool) -> list[dict]:
            if vec.size == 0:
                return []
            pos_idx = np.where(vec > 0)[0]
            if pos_idx.size == 0:
                return []
            vals = vec[pos_idx]
            k = min(10, pos_idx.size)
            top_idx_part = np.argpartition(vals, -k)[-k:]
            top_idx_sorted = top_idx_part[np.argsort(vals[top_idx_part])[::-1]]
            res: list[dict] = []
            maxv = float(vals[top_idx_sorted[0]]) if top_idx_sorted.size > 0 else 1.0
            maxv = max(maxv, 1e-6)
            for j in top_idx_sorted:
                idx = int(pos_idx[j])
                tag = vocab[idx] if idx < len(vocab) else None
                if not tag:
                    continue
                conf = float(vals[j]) / maxv
                if is_flavor:
                    res.append({"flavor": tag, "confidence": conf})
                else:
                    res.append({"mouthfeel": tag, "confidence": conf})
            return res

        flavors = top_k_from_slice(fv, flavors_vocab, True)
        mouthfeel = top_k_from_slice(mv, mouthfeel_vocab, False)
        return flavors, mouthfeel
    except Exception:
        return [], []

def _decode_age_bucket_from_row(row: np.ndarray) -> str | None:
    _compute_offsets_if_ready()
    if _OFFSETS is None or GLOBAL_VOCAB is None:
        return None
    try:
        start_a, len_a = _OFFSETS.get("age", (0, 0))
        if len_a <= 0:
            return None
        age_buckets = GLOBAL_VOCAB.get("age_buckets") or []
        if not age_buckets:
            return None
        a = row[start_a:start_a + len_a].astype(np.float32)
        # Inverse the weight approximately
        w_a = float(DEFAULT_WEIGHTS.get("age", 0.2) or 0.2)
        if w_a > 0:
            a = a / w_a
        if a.size == 0:
            return None
        idx = int(np.argmax(a))
        if 0 <= idx < len(age_buckets):
            return str(age_buckets[idx])
        return None
    except Exception:
        return None

# Attempt eager load (non-fatal if missing)
_load_vocab_and_catalog()

# Warm up the models on startup to avoid cold start penalty
print("🔥 Warming up models...")

# Warm up price and rating models first
try:
    warmup_data = {
        "variety": "Chardonnay",
        "country": "France", 
        "province": "Burgundy",
        "age": 3,
        "region_hierarchy": "France > Burgundy"
    }
    
    # Test price prediction
    price_warmup = predict_price_lite(warmup_data)
    print(f"✅ Price model warmed up successfully!")
    
    # Test rating prediction 
    rating_warmup_data = {
        **warmup_data,
        "price_min": float(price_warmup['weighted_lower']),
        "price_max": float(price_warmup['weighted_upper'])
    }
    rating_warmup = predict_rating_lite(rating_warmup_data)
    print(f"✅ Rating model warmed up successfully!")
    
except Exception as e:
    print(f"⚠️  Price/Rating model warmup failed: {e}")

# Warm up flavor prediction model - EAGERLY LOAD to avoid threading issues
try:
    print("🔍 Checking for flavor model files...")
    models_exist, missing_files = check_models_exist()
    
    if models_exist:
        print("✅ All flavor model files detected")
        print("🍷 Loading flavor model eagerly to avoid request-time delays...")
        
        # Load the model during startup to avoid threading issues
        success = load_model_eagerly()
        if success:
            print("✅ Flavor model loaded successfully during startup!")
            
            # Test a quick prediction to ensure everything works
            flavor_test_data = {
                **rating_warmup_data,
                "rating": float(rating_warmup['predicted_rating'])
            }
            test_flavors = predict_flavor_tags_from_dict(flavor_test_data, confidence_threshold=0.7, top_k=3)
            print(f"✅ Flavor model test successful! Found {len(test_flavors)} high-confidence flavors")
        else:
            print("⚠️  Flavor model loading failed during startup")
    else:
        print("⚠️  Some flavor model files not found:")
        for file in missing_files:
            print(f"    - {file}")
        print("    Note: Train the flavor model first with train_flavor_predictor.py")
        
except Exception as e:
    print(f"⚠️  Flavor model startup failed: {e}")
    print("    Note: Train the flavor model first with train_flavor_predictor.py")

# ---- Auth helpers and endpoints ----

def sign_app_jwt(payload: dict) -> str:
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_app_jwt(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  # type: ignore


def require_auth(fn):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
        if not token:
            return jsonify({"error": "Unauthorized"}), 401
        try:
            payload = decode_app_jwt(token)
            request.user = payload  # type: ignore
        except Exception:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


@app.route('/auth/google', methods=['POST'])
def auth_google():
    body = request.get_json(silent=True) or {}
    id_token_str = body.get('idToken')
    if not id_token_str:
        return jsonify({"error": "Missing idToken"}), 400
    try:
        ticket = google_id_token.verify_oauth2_token(id_token_str, google_requests.Request())
        if GOOGLE_CLIENT_IDS:
            aud = ticket.get('aud')
            if aud not in set(GOOGLE_CLIENT_IDS):
                return jsonify({"error": "Invalid audience"}), 401
        sub = ticket.get('sub')
        if not sub:
            return jsonify({"error": "Invalid token"}), 401
        email = ticket.get('email')
        name = ticket.get('name')
        picture = ticket.get('picture')

        user = db.users.find_one_and_update(
            {"google_id": sub},
            {
                "$setOnInsert": {"google_id": sub, "created_at": int(os.times().elapsed)},
                "$set": {"email": email, "name": name, "image": picture, "updated_at": int(os.times().elapsed)}
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )

        token = sign_app_jwt({"user_id": str(user.get("_id")), "google_id": sub})
        return jsonify({"token": token, "user": {"_id": str(user.get("_id")), "email": user.get("email"), "name": user.get("name"), "image": user.get("image")}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/me', methods=['GET'])
@require_auth
def me():
    user_id = request.user.get("user_id")  # type: ignore
    user = None
    try:
        if isinstance(user_id, str) and ObjectId.is_valid(user_id):
            user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None
    if not user:
        # Fallback via google_id if present
        gid = request.user.get("google_id") if isinstance(request.user, dict) else None  # type: ignore
        if gid:
            user = db.users.find_one({"google_id": gid})
    if not user:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"user": {"_id": str(user.get("_id")), "email": user.get("email"), "name": user.get("name"), "image": user.get("image")}})

@app.route('/me/profile', methods=['GET'])
@require_auth
def me_profile():
    user_id = request.user.get("user_id")  # type: ignore
    user = None
    try:
        if isinstance(user_id, str) and ObjectId.is_valid(user_id):
            user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None
    if not user:
        gid = request.user.get("google_id") if isinstance(request.user, dict) else None  # type: ignore
        if gid:
            user = db.users.find_one({"google_id": gid})
    if not user:
        return jsonify({"error": "Not found"}), 404
    vec = user.get("profile_vec")
    dim = user.get("profile_dim")
    updated = user.get("profile_updated_at")
    norm = None
    if isinstance(vec, list) and vec:
        try:
            import math
            norm = float(math.sqrt(sum((float(x) * float(x)) for x in vec)))
        except Exception:
            norm = None
    sample = vec[:8] if isinstance(vec, list) else []

    full = request.args.get('full') in ('1','true','yes')
    include_bottles = request.args.get('include_bottles') in ('1','true','yes')
    limit = min(int(request.args.get('limit', 100)), 500)

    resp = {
        "user_id": str(user.get("_id")),
        "has_profile": bool(isinstance(vec, list) and len(vec) == int(dim or 0) and (dim or 0) > 0),
        "profile_dim": int(dim) if dim is not None else None,
        "profile_updated_at": int(updated) if updated is not None else None,
        "norm": norm,
        "sample": sample,
    }
    if full and isinstance(vec, list):
        resp["profile_vec"] = vec
    if include_bottles:
        items = get_bottles_by_user(str(user.get("_id")), limit=limit)
        for it in items:
            if isinstance(it.get('_id'), ObjectId):
                it['_id'] = str(it['_id'])
        resp["bottles"] = items
    return jsonify(resp)

# ---- Cellar endpoints ----
@app.route('/cellar', methods=['GET'])
@require_auth
def get_cellar():
    user_id = request.user.get("user_id")  # type: ignore
    try:
        items = get_bottles_by_user(str(user_id), limit=int(request.args.get('limit', 100)))
        # Ensure _id serialized
        for it in items:
            if isinstance(it.get('_id'), ObjectId):
                it['_id'] = str(it['_id'])
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _compute_user_profile_vector(user_id: str) -> Tuple[List[float] | None, int | None]:
    """Compute a normalized profile vector by averaging embeddings of user's bottles.
    Returns (vec_list, dim) or (None, None) if not computable.
    """
    _load_vocab_and_catalog()
    try:
        bottles = get_bottles_by_user(str(user_id), limit=200)
        if not bottles:
            return None, None
        vecs: List[np.ndarray] = []
        weights: List[float] = []
        for b in bottles:
            try:
                wine = {
                    "name": b.get("name"),
                    "variety": b.get("variety"),
                    "country": b.get("country"),
                    "region1": b.get("region1"),
                    "region2": b.get("region2"),
                    "age": b.get("age"),
                    "price": (b.get("predicted", {}) or {}).get("price", {}).get("weighted_upper") or b.get("price"),
                    "rating": (b.get("predicted", {}) or {}).get("rating", {}).get("predicted_rating") or b.get("rating"),
                    "predicted": {
                        "flavors": (b.get("predicted", {}) or {}).get("flavors", []) or (b.get("tags", {}) or {}).get("flavors", []),
                        "mouthfeel": (b.get("predicted", {}) or {}).get("mouthfeel", []) or (b.get("tags", {}) or {}).get("mouthfeel", []),
                    },
                }
                vec, _ = build_embedding(
                    wine,
                    vocab=GLOBAL_VOCAB or {},
                    include_geo=True,
                    include_variety=True,
                    include_numeric=True,
                )
                v = np.asarray(vec, dtype=np.float32)
                if v.ndim != 1:
                    v = v.ravel()
                # Simple weight: favor higher ratings if available
                w = 1.0
                try:
                    r = float(wine.get("rating")) if wine.get("rating") is not None else None
                    if r is not None:
                        w = max(0.5, min(1.5, (r - 70.0) / 20.0))  # 70→0.5, 90→1.0, 100→1.5
                except Exception:
                    pass
                vecs.append(v)
                weights.append(float(w))
            except Exception:
                continue
        if not vecs:
            return None, None
        M = np.vstack(vecs)
        W = np.asarray(weights, dtype=np.float32)
        if W.sum() <= 0:
            W = np.ones_like(W)
        avg = (M * W[:, None]).sum(axis=0) / W.sum()
        # Normalize
        n = float(np.linalg.norm(avg))
        if n > 0:
            avg = avg / n
        vec_list = [float(x) for x in avg.astype(np.float32)]
        return vec_list, int(avg.shape[0])
    except Exception:
        return None, None

def _save_user_profile_vector(user_id: str, vec: List[float] | None, dim: int | None) -> None:
    try:
        q = None
        if isinstance(user_id, str) and ObjectId.is_valid(user_id):
            q = {"_id": ObjectId(user_id)}
        else:
            q = {"_id": user_id}
        if vec is None or dim is None:
            db.users.update_one(q, {"$unset": {"profile_vec": "", "profile_dim": ""}, "$set": {"profile_updated_at": now()}}, upsert=False)
        else:
            db.users.update_one(q, {"$set": {"profile_vec": vec, "profile_dim": int(dim), "profile_updated_at": now()}}, upsert=False)
    except Exception:
        pass

@app.route('/cellar', methods=['POST'])
@require_auth
def post_cellar():
    user_id = request.user.get("user_id")  # type: ignore
    body = request.get_json(silent=True) or {}
    try:
        public = bool(body.get('public', False))
        bottle = {k: v for k, v in body.items() if k not in ('_id', 'user_id', 'created_at')}
        bottle['public'] = public
        bottle_id = add_bottle(str(user_id), bottle)
        # Recompute and store profile vector
        vec, dim = _compute_user_profile_vector(str(user_id))
        _save_user_profile_vector(str(user_id), vec, dim)
        return jsonify({"ok": True, "bottle_id": bottle_id, "profile_updated": True, "profile_dim": dim})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cellar/public', methods=['GET'])
def list_public_cellar():
    try:
        items = get_public_bottles(limit=int(request.args.get('limit', 100)))
        for it in items:
            if isinstance(it.get('_id'), ObjectId):
                it['_id'] = str(it['_id'])
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_user_by_request() -> dict | None:
    user_id = request.user.get("user_id") if isinstance(request.user, dict) else None  # type: ignore
    user = None
    try:
        if isinstance(user_id, str) and ObjectId.is_valid(user_id):
            user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None
    if not user:
        gid = request.user.get("google_id") if isinstance(request.user, dict) else None  # type: ignore
        if gid:
            user = db.users.find_one({"google_id": gid})
    return user


@app.route('/recommend', methods=['POST'])
@require_auth
def recommend():
    """Return recommendations for the authenticated user based on their profile_vec.
    Body: {
      top_k?: int,
      diversity_lambda?: float (0..1),
      filters?: { variety?: str, country?: str, price_min?: number, price_max?: number },
      source?: 'catalog'|'public'|'both'
    }
    """
    _load_vocab_and_catalog()
    if CATALOG_X is None or CATALOG_META is None:
        return jsonify({"error": "Catalog not loaded"}), 500
    user = _load_user_by_request()
    if not user:
        return jsonify({"error": "User not found"}), 404
    vec = user.get("profile_vec")
    dim = user.get("profile_dim")
    if not isinstance(vec, list) or not vec or int(dim or 0) != CATALOG_X.shape[1]:
        return jsonify({"error": "Profile vector missing or dim mismatch"}), 400

    body = request.get_json(silent=True) or {}
    top_k = int(body.get('top_k', 10))
    top_k = max(1, min(top_k, 50))
    diversity_lambda = body.get('diversity_lambda', 0.2)
    try:
        diversity_lambda = float(diversity_lambda)
    except Exception:
        diversity_lambda = 0.2
    diversity_lambda = max(0.0, min(1.0, diversity_lambda))
    filters = body.get('filters') or {}
    f_variety = (filters.get('variety') or '').strip().lower() or None
    f_country = (filters.get('country') or '').strip().lower() or None
    f_price_min = filters.get('price_min')
    f_price_max = filters.get('price_max')
    source = (body.get('source') or 'both').lower()
    if source not in ('catalog', 'public', 'both'):
        source = 'both'
    blend = body.get('blend') or {}
    try:
        ratio_catalog = float(blend.get('ratio_catalog', 0.7))
    except Exception:
        ratio_catalog = 0.7
    ratio_catalog = max(0.0, min(1.0, ratio_catalog))

    # Build query vector
    q = np.asarray(vec, dtype=np.float32)
    if q.ndim != 1:
        q = q.ravel()
    # Defensive normalization
    n = float(np.linalg.norm(q))
    if n > 0:
        q = q / n

    matches_cat: list[dict] = []
    matches_pub: list[dict] = []

    # ---- Catalog branch ----
    if source in ('catalog', 'both'):
        N = CATALOG_X.shape[0]
        mask = np.ones(N, dtype=bool)
        if f_variety is not None:
            vmask = np.fromiter(((m.get('variety') or '') == f_variety for m in CATALOG_META), dtype=bool, count=N)
            mask &= vmask
        if f_country is not None:
            cmask = np.fromiter(((m.get('country') or '') == f_country for m in CATALOG_META), dtype=bool, count=N)
            mask &= cmask
        if f_price_min is not None or f_price_max is not None:
            pmin = float(f_price_min) if f_price_min is not None else -1e9
            pmax = float(f_price_max) if f_price_max is not None else 1e9
            pmask = np.fromiter((
                (m.get('price') is not None) and (float(m.get('price')) >= pmin) and (float(m.get('price')) <= pmax)
                for m in CATALOG_META
            ), dtype=bool, count=N)
            mask &= pmask
        scores_cat = (CATALOG_X.astype(np.float32) @ q)
        if not mask.all():
            scores_cat[~mask] = -1e9
        pool = max(top_k * 10, top_k)
        pool = min(pool, N)
        cand_idx = np.argpartition(scores_cat, -pool)[-pool:]
        cand_idx = cand_idx[np.argsort(scores_cat[cand_idx])[::-1]]
        # Optionally MMR rerank within catalog pool
        if diversity_lambda > 0 and top_k > 1 and len(cand_idx) > top_k:
            selected_indices: list[int] = []
            remaining = cand_idx.tolist()
            Xc = CATALOG_X[remaining].astype(np.float32)
            sims_to_query = scores_cat[remaining]
            best0 = int(np.argmax(sims_to_query))
            selected_indices.append(remaining[best0])
            Xsel = [Xc[best0]]
            del remaining[best0]
            Xc = np.delete(Xc, best0, axis=0)
            sims_to_query = np.delete(sims_to_query, best0, axis=0)
            while len(selected_indices) < top_k and len(remaining) > 0:
                if len(Xsel) == 1:
                    max_sim_sel = (Xc @ Xsel[0])
                else:
                    Xsel_mat = np.vstack(Xsel).T
                    max_sim_sel = np.max(Xc @ Xsel_mat, axis=1)
                mmr_scores = diversity_lambda * sims_to_query - (1.0 - diversity_lambda) * max_sim_sel
                j = int(np.argmax(mmr_scores))
                selected_indices.append(remaining[j])
                Xsel.append(Xc[j])
                del remaining[j]
                Xc = np.delete(Xc, j, axis=0)
                sims_to_query = np.delete(sims_to_query, j, axis=0)
            final_cat = np.array(selected_indices, dtype=int)
        else:
            final_cat = cand_idx[:top_k]
        for i in final_cat:
            m = CATALOG_META[int(i)] or {}
            # Decode tags from the embedding row
            try:
                row = CATALOG_X[int(i)].astype(np.float32)
            except Exception:
                row = None
            flv, mth = _decode_flavor_mouthfeel_from_row(row) if row is not None else ([], [])
            age_bucket = _decode_age_bucket_from_row(row) if row is not None else None
            pred_block = {
                "rating": {"predicted_rating": m.get("rating")},
                "price": {"weighted_lower": m.get("price"), "weighted_upper": m.get("price")},
                "flavors": flv,
                "mouthfeel": mth,
                "age_bucket": age_bucket,
            }
            matches_cat.append({
                "index": int(i),
                "score": float(scores_cat[int(i)]),
                "name": m.get("name"),
                "variety": m.get("variety"),
                "country": m.get("country"),
                "region1": m.get("region1"),
                "region2": m.get("region2"),
                "age": m.get("age"),
                "price": m.get("price"),
                "rating": m.get("rating"),
                "predicted": pred_block,
                "because": _build_because(m, f_variety, f_country, f_price_min, f_price_max),
                "source": "catalog",
            })

    # ---- Public branch ----
    if source in ('public', 'both'):
        # Cap how many public bottles we consider for performance
        cap = min(int(body.get('public_limit', 300)), 1000)
        q_public = {"public": True}
        if f_variety is not None:
            q_public["variety"] = f_variety
        if f_country is not None:
            q_public["country"] = f_country
        public_bottles = list(db.bottles.find(q_public).sort("created_at", -1).limit(cap))
        if public_bottles:
            X_list: list[np.ndarray] = []
            meta_list: list[dict] = []
            for b in public_bottles:
                try:
                    wine = {
                        "name": b.get("name"),
                        "variety": b.get("variety"),
                        "country": b.get("country"),
                        "region1": b.get("region1"),
                        "region2": b.get("region2"),
                        "age": b.get("age"),
                        "price": b.get("price") or (b.get("predicted", {}) or {}).get("price", {}).get("weighted_upper"),
                        "rating": b.get("rating") or (b.get("predicted", {}) or {}).get("rating", {}).get("predicted_rating"),
                        "predicted": {
                            "flavors": (b.get("predicted", {}) or {}).get("flavors", []) or (b.get("tags", {}) or {}).get("flavors", []),
                            "mouthfeel": (b.get("predicted", {}) or {}).get("mouthfeel", []) or (b.get("tags", {}) or {}).get("mouthfeel", []),
                        },
                    }
                    vec_b, _ = build_embedding(
                        wine,
                        vocab=GLOBAL_VOCAB or {},
                        include_geo=True,
                        include_variety=True,
                        include_numeric=True,
                    )
                    vb = np.asarray(vec_b, dtype=np.float32)
                    if vb.ndim != 1:
                        vb = vb.ravel()
                    nvb = float(np.linalg.norm(vb))
                    if nvb > 0:
                        vb = vb / nvb
                    # Decode age bucket from embedding row
                    age_bucket = _decode_age_bucket_from_row(vb)
                    # Price filter for public
                    if f_price_min is not None or f_price_max is not None:
                        pmin = float(f_price_min) if f_price_min is not None else -1e9
                        pmax = float(f_price_max) if f_price_max is not None else 1e9
                        pv = wine.get("price")
                        if pv is None or float(pv) < pmin or float(pv) > pmax:
                            continue
                    X_list.append(vb)
                    pb_pred = (b.get("predicted") or {}).copy() if isinstance(b.get("predicted"), dict) else {}
                    if age_bucket and isinstance(pb_pred, dict):
                        pb_pred["age_bucket"] = age_bucket
                    meta_list.append({
                        "_id": str(b.get("_id")) if isinstance(b.get("_id"), ObjectId) else b.get("_id"),
                        "name": wine.get("name"),
                        "variety": wine.get("variety"),
                        "country": wine.get("country"),
                        "region1": wine.get("region1"),
                        "region2": wine.get("region2"),
                        "age": wine.get("age"),
                        "price": wine.get("price"),
                        "rating": wine.get("rating"),
                        "predicted": pb_pred,
                    })
                except Exception:
                    continue
            if X_list:
                Xpub = np.vstack(X_list)  # M x D
                scores_pub = (Xpub @ q)
                # take top_k from public set
                m = min(top_k, scores_pub.shape[0])
                idx_pub = np.argpartition(scores_pub, -m)[-m:]
                idx_pub = idx_pub[np.argsort(scores_pub[idx_pub])[::-1]]
                for j in idx_pub:
                    mta = meta_list[int(j)]
                    matches_pub.append({
                        "index": -1,  # not from catalog
                        "score": float(scores_pub[int(j)]),
                        "name": mta.get("name"),
                        "variety": mta.get("variety"),
                        "country": mta.get("country"),
                        "region1": mta.get("region1"),
                        "region2": mta.get("region2"),
                        "age": mta.get("age"),
                        "price": mta.get("price"),
                        "rating": mta.get("rating"),
                        "predicted": mta.get("predicted"),
                        "because": _build_because(mta, f_variety, f_country, f_price_min, f_price_max),
                        "source": "public",
                        "bottle_id": mta.get("_id"),
                    })

    # Combine with blending
    if not matches_cat and not matches_pub:
        return jsonify({"top_k": top_k, "diversity_lambda": diversity_lambda, "filters": {"variety": f_variety, "country": f_country, "price_min": f_price_min, "price_max": f_price_max}, "matches": []})
    # Sort each source by score
    if matches_cat:
        matches_cat.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)
    if matches_pub:
        matches_pub.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)
    # Desired counts per source
    want_cat = int(round(top_k * ratio_catalog)) if source == 'both' else (top_k if source == 'catalog' else 0)
    want_pub = top_k - want_cat if source == 'both' else (top_k if source == 'public' else 0)
    out: list[dict] = []
    i_cat = 0
    i_pub = 0
    # Interleave roughly by ratio
    while len(out) < top_k and (i_cat < len(matches_cat) or i_pub < len(matches_pub)):
        take_cat = len(out) % 2 == 0  # alternate starting with catalog
        if source == 'public':
            take_cat = False
        if source == 'catalog':
            take_cat = True
        if take_cat and i_cat < len(matches_cat) and (len([m for m in out if m.get('source')=='catalog']) < want_cat or i_pub >= len(matches_pub)):
            out.append(matches_cat[i_cat]); i_cat += 1; continue
        if i_pub < len(matches_pub) and (len([m for m in out if m.get('source')=='public']) < want_pub or i_cat >= len(matches_cat)):
            out.append(matches_pub[i_pub]); i_pub += 1; continue
        # Fallback fill
        if i_cat < len(matches_cat):
            out.append(matches_cat[i_cat]); i_cat += 1; continue
        if i_pub < len(matches_pub):
            out.append(matches_pub[i_pub]); i_pub += 1; continue
        break
    out = out[:top_k]

    return jsonify({
        "top_k": int(top_k),
        "diversity_lambda": float(diversity_lambda),
        "filters": {"variety": f_variety, "country": f_country, "price_min": f_price_min, "price_max": f_price_max},
        "source": source,
        "blend": {"ratio_catalog": ratio_catalog},
        "matches": out
    })


def _build_because(m: dict, f_variety, f_country, f_price_min, f_price_max) -> str:
    parts: list[str] = []
    if f_variety and (m.get('variety') == f_variety):
        parts.append(f"same variety: {f_variety}")
    if f_country and (m.get('country') == f_country):
        parts.append(f"same country: {f_country}")
    try:
        p = float(m.get('price')) if m.get('price') is not None else None
        if p is not None and f_price_min is not None and f_price_max is not None:
            parts.append("within price range")
    except Exception:
        pass
    if not parts:
        parts.append("high similarity to your profile")
    return ", ".join(parts)

@app.route('/match-real', methods=['POST'])
def match_real():
    """Return top-k nearest catalog indices and scores for the given wine payload.
    Expects the same wine shape as /embed. Uses normalized vocab and catalog embeddings.
    """
    _load_vocab_and_catalog()
    if CATALOG_X is None or GLOBAL_VOCAB is None:
        return jsonify({"error": "Catalog or vocab not loaded on server"}), 500
    body = request.get_json(silent=True) or {}
    top_k = int(body.get('top_k', 5))
    try:
        vec, _ = build_embedding(
            body,
            vocab=GLOBAL_VOCAB,
            include_geo=True,
            include_variety=True,
            include_numeric=True,
        )
        q = np.asarray(vec, dtype=np.float32)
        if q.ndim != 1:
            q = q.ravel()
        if CATALOG_X.shape[1] != q.shape[0]:
            return jsonify({"error": f"Dim mismatch: catalog {CATALOG_X.shape[1]} vs query {q.shape[0]}"}), 400
        # Ensure unit length (defensive)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        # Cosine scores via dot product (CATALOG_X is L2-normalized)
        scores = (CATALOG_X.astype(np.float32) @ q)
        if top_k <= 0:
            top_k = 5
        top_k = min(int(top_k), scores.shape[0])
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        result = []
        for i in idx:
            item = {"index": int(i), "score": float(scores[i])}
            if CATALOG_META and 0 <= int(i) < len(CATALOG_META):
                m = CATALOG_META[int(i)] or {}
                item.update({
                    "name": m.get("name"),
                    "variety": m.get("variety"),
                    "country": m.get("country"),
                    "region1": m.get("region1"),
                    "region2": m.get("region2"),
                    "price": m.get("price"),
                    "rating": m.get("rating"),
                })
            result.append(item)
        return jsonify({"count": int(CATALOG_X.shape[0]), "dim": int(CATALOG_X.shape[1]), "top_k": top_k, "matches": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-price-lite', methods=['POST'])
def predict_price_lite_endpoint():
    data = request.json
    print(f"Received data (price): {data}")
    try:
        prediction = predict_price_lite(data)
        return jsonify(prediction)
    except Exception as e:
        print(f"Error (price): {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-rating-lite', methods=['POST'])
def predict_rating_lite_endpoint():
    data = request.json
    print(f"Received data (rating): {data}")

    try:
        # Step 1: Predict price bounds from user input
        price_prediction = predict_price_lite(data)
        price_min = float(price_prediction['weighted_lower'])
        price_max = float(price_prediction['weighted_upper'])

        # Step 2: Add these to input and call rating model
        full_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max
        }

        # Step 3: Predict rating
        rating_prediction = predict_rating_lite(full_input)
        return jsonify(rating_prediction)

    except Exception as e:
        print(f"Error (rating): {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-flavor', methods=['POST'])
def predict_flavor_endpoint():
    data = request.json
    print(f"Received data (flavor): {data}")
    
    try:
        # Extract parameters from request
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check if we have all required basic fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # If price_min/price_max not provided, predict them
        if 'price_min' not in data or 'price_max' not in data:
            print("  Predicting prices first...")
            price_prediction = predict_price_lite(data)
            data['price_min'] = float(price_prediction['weighted_lower'])
            data['price_max'] = float(price_prediction['weighted_upper'])
        
        # If rating not provided, predict it
        if 'rating' not in data:
            print("  Predicting rating first...")
            rating_prediction = predict_rating_lite(data)
            data['rating'] = float(rating_prediction['predicted_rating'])
        
        # Now predict flavors
        print("  Predicting flavors...")
        flavor_prediction = predict_flavor_tags_from_dict(
            data,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        return jsonify({
            "flavors": flavor_prediction,
            "input_data": {
                "variety": data['variety'],
                "country": data['country'],
                "province": data['province'],
                "age": data['age'],
                "region_hierarchy": data['region_hierarchy'],
                "price_min": data['price_min'],
                "price_max": data['price_max'],
                "rating": data['rating']
            },
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k
            }
        })

    except Exception as e:
        print(f"Error (flavor): {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-mouthfeel', methods=['POST'])
def predict_mouthfeel_endpoint():
    data = request.json
    print(f"Received data (mouthfeel): {data}")
    
    try:
        # Extract parameters from request
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check if we have all required basic fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # If price_min/price_max not provided, predict them
        if 'price_min' not in data or 'price_max' not in data:
            print("  Predicting prices first...")
            price_prediction = predict_price_lite(data)
            data['price_min'] = float(price_prediction['weighted_lower'])
            data['price_max'] = float(price_prediction['weighted_upper'])
        
        # If rating not provided, predict it
        if 'rating' not in data:
            print("  Predicting rating first...")
            rating_prediction = predict_rating_lite(data)
            data['rating'] = float(rating_prediction['predicted_rating'])
        
        # Now predict mouthfeel
        print("  Predicting mouthfeel...")
        mouthfeel_prediction = predict_mouthfeel_tags_from_dict(
            data,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        result = {
            "mouthfeel": mouthfeel_prediction,
            "input_data": data,
            "prediction_info": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k,
                "total_found": len(mouthfeel_prediction)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in mouthfeel prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-all', methods=['POST'])
def predict_all_endpoint():
    """Convenience endpoint that predicts price, rating, flavors, and mouthfeel all at once"""
    data = request.json
    print(f"Received data (all): {data}")
    
    try:
        # Extract prediction parameters
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check required fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Step 1: Predict price
        print("  Step 1: Predicting price...")
        price_prediction = predict_price_lite(data)
        price_min = float(price_prediction['weighted_lower'])
        price_max = float(price_prediction['weighted_upper'])
        
        # Step 2: Predict rating (using price predictions)
        print("  Step 2: Predicting rating...")
        rating_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max
        }
        rating_prediction = predict_rating_lite(rating_input)
        rating = float(rating_prediction['predicted_rating'])
        
        # Step 3: Predict flavors (using price and rating predictions)
        print("  Step 3: Predicting flavors...")
        prediction_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max,
            "rating": rating
        }
        flavor_prediction = predict_flavor_tags_from_dict(
            prediction_input,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Step 4: Predict mouthfeel (using price and rating predictions)
        print("  Step 4: Predicting mouthfeel...")
        mouthfeel_prediction = predict_mouthfeel_tags_from_dict(
            prediction_input,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Combine all results
        result = {
            "price": price_prediction,
            "rating": rating_prediction,
            "flavors": flavor_prediction,
            "mouthfeel": mouthfeel_prediction,
            "input_data": data,
            "prediction_info": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k,
                "flavor_count": len(flavor_prediction),
                "mouthfeel_count": len(mouthfeel_prediction)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in combined prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed_wine():
    body = request.get_json(silent=True) or {}
    try:
        vocab = body.get('vocab') if isinstance(body.get('vocab'), dict) else None
        weights = body.get('weights') if isinstance(body.get('weights'), dict) else None
        wine = {k: v for k, v in body.items() if k not in ('vocab','weights')}
        vec, vocab_used = build_embedding(wine, vocab=vocab, weights=weights)
        return jsonify({
            "embedding": vec,
            "length": len(vec),
            "vocab": vocab_used,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mongo_status = "connected"
    try:
        db.command('ping')
    except Exception as e:
        mongo_status = f"error: {e.__class__.__name__}"

    return jsonify({
        "status": "healthy",
        "models": {
            "price": "available",
            "rating": "available", 
            "flavor": "available" if 'predict_flavor_tags_from_dict' in globals() else "training_required"
        },
        "mongo": {
            "status": mongo_status,
            "ready_at_startup": bool(MONGO_READY)
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Available endpoints:")
    print("   POST /auth/google        - Verify Google ID token and return app JWT")
    print("   GET  /me                 - Get current user by JWT")
    print("   POST /predict-price-lite   - Predict wine price range")
    print("   POST /predict-rating-lite  - Predict wine rating")
    print("   POST /predict-flavor       - Predict wine flavors")
    print("   POST /predict-all          - Predict price, rating, and flavors")
    print("   GET  /health               - Health check")
    app.run(debug=True, host='0.0.0.0', port=5001)

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math

# Default weights for blocks
DEFAULT_WEIGHTS = {
    "flavors": 1.0,
    "mouthfeel": 0.7,
    "variety": 0.4,
    "country": 0.2,
    "region1": 0.2,
    "region2": 0.1,
    "age": 0.2,
    "rating": 0.2,
    "price": 0.2,
}

def _index_map(vocab: List[str]) -> Dict[str, int]:
    return {k: i for i, k in enumerate(vocab)}

def _one_hot(value: str | None, index: Dict[str, int], length: int) -> List[float]:
    vec = [0.0] * length
    if value is not None and value in index:
        vec[index[value]] = 1.0
    return vec

def _scale_01(x: Any, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (xf - lo) / (hi - lo)))

def _bucket_price(p: Any) -> str:
    try:
        v = float(p)
    except Exception:
        return "unknown"
    if v < 15: return "<15"
    if v < 30: return "15-30"
    if v < 60: return "30-60"
    return "60+"

def _bucket_from_scalar(x: Any, buckets: List[str]) -> str:
    # buckets like ["<80","80-85","85-90","90-95","95+"]
    try:
        v = float(x)
    except Exception:
        return buckets[-1] if buckets else "unknown"
    # simple parse: assume formats like "a-b" or "<a" or "b+" and pick matching
    for b in buckets:
        b = str(b)
        if '-' in b:
            lo, hi = b.split('-', 1)
            try:
                lo_v = float(lo)
                hi_v = float(hi)
                if lo_v <= v <= hi_v:
                    return b
            except Exception:
                continue
        elif b.startswith('<'):
            try:
                th = float(b[1:])
                if v < th:
                    return b
            except Exception:
                continue
        elif b.endswith('+'):
            try:
                th = float(b[:-1])
                if v >= th:
                    return b
            except Exception:
                continue
    return buckets[-1] if buckets else "unknown"

def build_embedding(
    wine: Dict[str, Any],
    vocab: Dict[str, List[str]] | Dict[str, Any] | None = None,
    weights: Dict[str, float] | None = None,
    include_geo: bool = True,
    include_variety: bool = True,
    include_numeric: bool = True,
) -> Tuple[List[float], Dict[str, List[str]]]:
    """Build a normalized embedding for a wine.

    Supports both legacy vocab keys (flavors, mouthfeel, varieties, countries, regions1, regions2, price_bins)
    and v2 keys (flavor_vocab, mouthfeel_vocab, geography_vocab{countries,primary_regions,secondary_regions}, variety_vocab,
    price_buckets, rating_buckets, age_buckets).

    Set include_geo/include_variety/include_numeric to False to omit those blocks entirely (zero length).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    vocab = vocab or {}

    # Resolve vocab (support v1 or v2 schema)
    flavors_vocab = vocab.get("flavors") or vocab.get("flavor_vocab") or []
    mouthfeel_vocab = vocab.get("mouthfeel") or vocab.get("mouthfeel_vocab") or []

    # Geography
    geo = vocab.get("geography_vocab") or {}
    countries_vocab = []
    regions1_vocab = []
    regions2_vocab = []
    if include_geo:
        countries_vocab = vocab.get("countries") or geo.get("countries") or []
        regions1_vocab = vocab.get("regions1") or geo.get("primary_regions") or []
        regions2_vocab = vocab.get("regions2") or geo.get("secondary_regions") or []

    # Varieties
    varieties_vocab = vocab.get("varieties") or vocab.get("variety_vocab") or []
    if not include_variety:
        varieties_vocab = []

    # Numeric buckets
    price_bins = vocab.get("price_bins") or vocab.get("price_buckets") or []
    rating_bins = vocab.get("rating_buckets") or []
    age_bins = vocab.get("age_buckets") or []
    if not include_numeric:
        price_bins = []
        rating_bins = []
        age_bins = []

    # Collect tags
    predicted = wine.get("predicted", {}) if isinstance(wine.get("predicted", {}), dict) else {}
    flavors_found = [t for t in (predicted.get("flavors") or []) if isinstance(t, dict) and t.get("flavor")]
    mouthfeel_found = [t for t in (predicted.get("mouthfeel") or []) if isinstance(t, dict) and t.get("mouthfeel")]

    # Build indices
    flavors_index = _index_map(flavors_vocab)
    mouthfeel_index = _index_map(mouthfeel_vocab)
    varieties_index = _index_map(varieties_vocab)
    countries_index = _index_map(countries_vocab)
    regions1_index = _index_map(regions1_vocab)
    regions2_index = _index_map(regions2_vocab)
    price_index = _index_map(price_bins)
    rating_index = _index_map(rating_bins)
    age_index = _index_map(age_bins)

    # Flavors block
    f_vec = [0.0] * len(flavors_vocab)
    for t in flavors_found:
        idx = flavors_index.get(t["flavor"])  # type: ignore
        if idx is not None:
            try:
                conf = float(t.get("confidence", 1.0))
            except Exception:
                conf = 1.0
            f_vec[idx] = max(f_vec[idx], max(0.0, min(1.0, conf)))

    # Mouthfeel block
    m_vec = [0.0] * len(mouthfeel_vocab)
    for t in mouthfeel_found:
        idx = mouthfeel_index.get(t["mouthfeel"])  # type: ignore
        if idx is not None:
            try:
                conf = float(t.get("confidence", 1.0))
            except Exception:
                conf = 1.0
            m_vec[idx] = max(m_vec[idx], max(0.0, min(1.0, conf)))

    # Variety / Country / Regions one-hot
    v_vec: List[float] = []
    if varieties_vocab:
        v_vec = _one_hot(wine.get("variety"), varieties_index, len(varieties_vocab))
    c_vec: List[float] = []
    if countries_vocab:
        c_vec = _one_hot(wine.get("country"), countries_index, len(countries_vocab))
    r1_vec: List[float] = []
    if regions1_vocab:
        r1_vec = _one_hot(wine.get("region1") or wine.get("province"), regions1_index, len(regions1_vocab))
    r2_vec: List[float] = []
    if regions2_vocab:
        r2_vec = _one_hot(wine.get("region2"), regions2_index, len(regions2_vocab))

    # Numeric buckets one-hot (only if bins provided)
    price_vec: List[float] = []
    if price_bins:
        bucket = _bucket_from_scalar(wine.get("price"), price_bins)
        price_vec = _one_hot(bucket, price_index, len(price_bins))
    rating_vec: List[float] = []
    if rating_bins:
        # Use predicted.rating.predicted_rating or plain rating
        pr_val = None
        if isinstance(predicted, dict) and isinstance(predicted.get("rating"), dict):
            pr_val = predicted.get("rating", {}).get("predicted_rating")
        else:
            pr_val = wine.get("rating")
        bucket = _bucket_from_scalar(pr_val, rating_bins)
        rating_vec = _one_hot(bucket, rating_index, len(rating_bins))
    age_vec: List[float] = []
    if age_bins:
        bucket = _bucket_from_scalar(wine.get("age"), age_bins)
        age_vec = _one_hot(bucket, age_index, len(age_bins))

    # Concatenate with weights
    concat: List[float] = []
    def _append(block: List[float], scale: float):
        if not block:
            return
        if scale == 1.0:
            concat.extend(block)
        else:
            concat.extend([scale * x for x in block])

    _append(f_vec, w["flavors"])        # flavors
    _append(m_vec, w["mouthfeel"])      # mouthfeel
    _append(v_vec, w["variety"])        # variety
    _append(c_vec, w["country"])        # country
    _append(r1_vec, w["region1"])       # region1
    _append(r2_vec, w["region2"])       # region2
    # numeric blocks have their own weights
    _append(age_vec, w["age"])          # age bucket
    _append(rating_vec, w["rating"])    # rating bucket
    _append(price_vec, w["price"])      # price bucket

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in concat)) or 1.0
    vec = [x / norm for x in concat]

    vocab_used = {
        "flavors": flavors_vocab,
        "mouthfeel": mouthfeel_vocab,
        "varieties": varieties_vocab,
        "countries": countries_vocab,
        "regions1": regions1_vocab,
        "regions2": regions2_vocab,
        "price_bins": price_bins,
        "rating_buckets": rating_bins,
        "age_buckets": age_bins,
    }
    return vec, vocab_used 
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

def build_embedding(
    wine: Dict[str, Any],
    vocab: Dict[str, List[str]] | None = None,
    weights: Dict[str, float] | None = None,
) -> Tuple[List[float], Dict[str, List[str]]]:
    """Build a normalized embedding for a wine.

    wine: expects fields
      - predicted.flavors: [{flavor, confidence}]
      - predicted.mouthfeel: [{mouthfeel, confidence}]
      - variety, country, region1?, region2?, age, predicted.rating.predicted_rating, price (optional)
    vocab: optional fixed vocabularies with keys: flavors, mouthfeel, varieties, countries, regions1, regions2, price_bins
    weights: optional override of block weights

    Returns (vector, vocab_used)
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Vocab defaults (kept small; can be overridden by client)
    default_flavors = vocab.get("flavors") if vocab else None
    default_mouthfeel = vocab.get("mouthfeel") if vocab else None
    default_varieties = vocab.get("varieties") if vocab else ["Cabernet Sauvignon","Pinot Noir","Chardonnay","Sauvignon Blanc","Syrah","Merlot","Riesling","Zinfandel","Malbec","Sangiovese","Other"]
    default_countries = vocab.get("countries") if vocab else ["US","France","Italy","Spain","Portugal","Chile","Argentina","Germany","Austria","Australia","Other"]
    default_regions1 = vocab.get("regions1") if vocab else None
    default_regions2 = vocab.get("regions2") if vocab else None
    default_price_bins = vocab.get("price_bins") if vocab else ["<15","15-30","30-60","60+","unknown"]

    # Collect tags found if vocab for tags not provided
    flavors_found = [t for t in (wine.get("predicted",{}).get("flavors") or []) if isinstance(t, dict) and t.get("flavor")]
    mouthfeel_found = [t for t in (wine.get("predicted",{}).get("mouthfeel") or []) if isinstance(t, dict) and t.get("mouthfeel")]

    flavors_vocab = default_flavors or sorted({t["flavor"] for t in flavors_found})
    mouthfeel_vocab = default_mouthfeel or sorted({t["mouthfeel"] for t in mouthfeel_found})

    flavors_index = _index_map(flavors_vocab)
    mouthfeel_index = _index_map(mouthfeel_vocab)
    varieties_index = _index_map(default_varieties)
    countries_index = _index_map(default_countries)
    regions1_vocab = default_regions1 or ([] if wine.get("region1") is None else [str(wine.get("region1"))])
    regions2_vocab = default_regions2 or ([] if wine.get("region2") is None else [str(wine.get("region2"))])
    regions1_index = _index_map(regions1_vocab)
    regions2_index = _index_map(regions2_vocab)
    price_bins_index = _index_map(default_price_bins)

    # Flavors block
    f_vec = [0.0] * len(flavors_vocab)
    for t in flavors_found:
        idx = flavors_index.get(t["flavor"])  # type: ignore
        if idx is not None:
            try:
                conf = float(t.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            f_vec[idx] = max(f_vec[idx], max(0.0, min(1.0, conf)))

    # Mouthfeel block
    m_vec = [0.0] * len(mouthfeel_vocab)
    for t in mouthfeel_found:
        idx = mouthfeel_index.get(t["mouthfeel"])  # type: ignore
        if idx is not None:
            try:
                conf = float(t.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            m_vec[idx] = max(m_vec[idx], max(0.0, min(1.0, conf)))

    # Variety / Country / Regions one-hot
    v_vec = _one_hot(wine.get("variety"), varieties_index, len(default_varieties))
    c_vec = _one_hot(wine.get("country"), countries_index, len(default_countries))
    r1_vec = _one_hot(wine.get("region1"), regions1_index, len(regions1_vocab)) if regions1_vocab else []
    r2_vec = _one_hot(wine.get("region2"), regions2_index, len(regions2_vocab)) if regions2_vocab else []

    # Age / Rating scalar, Price bucket one-hot
    age01 = _scale_01(wine.get("age"), 0, 30)  # cap at 30 years
    pr = wine.get("predicted", {}).get("rating") if isinstance(wine.get("predicted", {}), dict) else None
    rating_val = pr.get("predicted_rating") if isinstance(pr, dict) else None
    rating01 = _scale_01(rating_val, 70, 100) if rating_val is not None else 0.0

    price_bucket = _bucket_price(wine.get("price"))
    p_vec = _one_hot(price_bucket, price_bins_index, len(default_price_bins))

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
    concat.append(w["age"] * age01)     # age scalar
    concat.append(w["rating"] * rating01)  # rating scalar
    _append(p_vec, w["price"])          # price bucket

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in concat)) or 1.0
    vec = [x / norm for x in concat]

    vocab_used = {
        "flavors": flavors_vocab,
        "mouthfeel": mouthfeel_vocab,
        "varieties": default_varieties,
        "countries": default_countries,
        "regions1": regions1_vocab,
        "regions2": regions2_vocab,
        "price_bins": default_price_bins,
    }
    return vec, vocab_used 
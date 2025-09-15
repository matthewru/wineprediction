#!/usr/bin/env python3
import csv, json, sys, argparse, time, unicodedata, re
from typing import List, Dict, Any
import requests
# Ensure backend package root is on sys.path for local import
import os as _os
_sys_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)
from embedding_utils import build_embedding

DEFAULT_TAG_SPLIT_RE = re.compile(r'[;,]')


def parse_args():
    p = argparse.ArgumentParser(description='Build catalog embeddings from CSV via /embed endpoint or locally')
    p.add_argument('--csv', required=True, help='Path to wine_clean_tagged.csv')
    p.add_argument('--api', default='http://localhost:5001', help='Backend base URL (default: http://localhost:5001)')
    p.add_argument('--out', default='backend/data/catalog_wines.jsonl', help='Output JSONL path')
    # Defaults set to wine_clean_tagged.csv columns
    p.add_argument('--flavor-col', default='flavor_tags', help='CSV column with flavors (comma-separated)')
    p.add_argument('--mouthfeel-col', default='mouthfeel_tags', help='CSV column with mouthfeel (comma-separated)')
    p.add_argument('--name-col', default='title', help='CSV column for wine name')
    p.add_argument('--variety-col', default='variety', help='CSV column for variety')
    p.add_argument('--country-col', default='country', help='CSV column for country')
    p.add_argument('--region1-col', default='province', help='CSV column for primary region')
    p.add_argument('--region2-col', default='region_hierarchy', help='CSV column that contains region hierarchy; we will use its LAST token')
    p.add_argument('--age-col', default='age', help='CSV column for age')
    p.add_argument('--price-col', default='price', help='CSV column for real price')
    p.add_argument('--rating-col', default='points', help='CSV column for real rating')
    p.add_argument('--vocab-json', default=None, help='Optional JSON file with fixed vocab (v2 schema)')
    p.add_argument('--progress-every', type=int, default=1000, help='Print progress every N rows')
    p.add_argument('--retries', type=int, default=2, help='Retries on /embed failure (default: 2)')
    p.add_argument('--retry-wait', type=float, default=0.2, help='Seconds to wait between retries (default: 0.2)')
    p.add_argument('--local', action='store_true', help='Compute embeddings locally (no HTTP)')
    return p.parse_args()


def _norm(s: str) -> str:
    s = (s or '').strip()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s.lower()


def parse_tag_list(raw: str) -> List[str]:
    raw = (raw or '').strip()
    if not raw:
        return []
    if raw.startswith('[') and raw.endswith(']'):
        try:
            arr = json.loads(raw)
            return [_norm(str(x)) for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = [t.strip().strip('"\'') for t in DEFAULT_TAG_SPLIT_RE.split(raw) if t.strip()]
    return [_norm(t) for t in parts]


def last_region_token(h: str | None) -> str | None:
    if not h:
        return None
    parts = [p.strip() for p in h.split('>') if p.strip()]
    return parts[-1] if parts else None


def main():
    args = parse_args()
    vocab = None
    if args.vocab_json:
        with open(args.vocab_json, 'r') as vf:
            vocab = json.load(vf)

    total = 0
    written = 0
    failed = 0

    with open(args.csv, 'r', newline='', encoding='utf-8') as f, open(args.out, 'w', encoding='utf-8') as out:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                name = row.get(args.name_col) or ''
                variety = _norm(row.get(args.variety_col) or '') or None
                country = _norm(row.get(args.country_col) or '') or None
                region1 = _norm(row.get(args.region1_col) or '') or None
                # region2 source is a hierarchy => take LAST term, normalize
                region_h = row.get(args.region2_col) or None
                region2 = _norm(last_region_token(region_h) or '') or None
                age = row.get(args.age_col) or None
                price = row.get(args.price_col) or None
                rating = row.get(args.rating_col) or None

                flavors = parse_tag_list(row.get(args.flavor_col) or '')
                mouthfeel = parse_tag_list(row.get(args.mouthfeel_col) or '')

                payload: Dict[str, Any] = {
                    "name": name,
                    "variety": variety,
                    "country": country,
                    "region1": region1,
                    "region2": region2,
                    "age": float(age) if age not in (None, '') else None,
                    "price": float(price) if price not in (None, '') else None,
                    "predicted": {
                        "rating": {"predicted_rating": float(rating)} if rating not in (None, '') else {},
                        "flavors": [{"flavor": t, "confidence": 1.0} for t in flavors],
                        "mouthfeel": [{"mouthfeel": t, "confidence": 1.0} for t in mouthfeel],
                    }
                }
                if vocab:
                    payload['vocab'] = vocab

                if args.local:
                    vec, _ = build_embedding(payload, vocab=vocab, include_geo=True, include_variety=True, include_numeric=True)
                    rec = {
                        "name": name,
                        "variety": variety,
                        "country": country,
                        "region1": region1,
                        "region2": region2,
                        "price": payload.get("price"),
                        "rating": (payload.get("predicted", {}) or {}).get("rating", {}).get("predicted_rating"),
                        "tags": {
                            "flavors": flavors,
                            "mouthfeel": mouthfeel,
                        },
                        "embedding": vec,
                        "embedding_len": len(vec),
                    }
                    out.write(json.dumps(rec) + "\n")
                    written += 1
                else:
                    # HTTP mode with retry
                    attempt = 0
                    while True:
                        attempt += 1
                        resp = requests.post(f"{args.api}/embed", json=payload, timeout=30)
                        if resp.ok:
                            break
                        if attempt > args.retries + 1:
                            break
                        time.sleep(args.retry_wait)
                    if not resp.ok:
                        failed += 1
                        sys.stderr.write(f"Embed failed for row {total} ({name}): {resp.status_code} {resp.text}\n")
                        continue

                    data = resp.json()
                    rec = {
                        "name": name,
                        "variety": variety,
                        "country": country,
                        "region1": region1,
                        "region2": region2,
                        "price": payload.get("price"),
                        "rating": (payload.get("predicted", {}) or {}).get("rating", {}).get("predicted_rating"),
                        "tags": {
                            "flavors": flavors,
                            "mouthfeel": mouthfeel,
                        },
                        "embedding": data.get("embedding"),
                        "embedding_len": data.get("length"),
                    }
                    out.write(json.dumps(rec) + "\n")
                    written += 1

                if written % args.progress_every == 0:
                    print(f"Processed {total} rows, written {written}, failed {failed}")
            except Exception as e:
                failed += 1
                sys.stderr.write(f"Row {total} failed: {e}\n")
                continue

    print(f"Done. Total={total}, written={written}, failed={failed}, output={args.out}")

if __name__ == '__main__':
    main() 
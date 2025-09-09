#!/usr/bin/env python3
import csv, json, sys, argparse
from typing import List, Dict, Any
import requests

DEFAULT_TAG_SPLIT = ';'


def parse_args():
    p = argparse.ArgumentParser(description='Build catalog embeddings from CSV via /embed endpoint')
    p.add_argument('--csv', required=True, help='Path to wine_clean_tagged.csv')
    p.add_argument('--api', default='http://localhost:5001', help='Backend base URL (default: http://localhost:5001)')
    p.add_argument('--out', default='backend/data/catalog_wines.jsonl', help='Output JSONL path')
    p.add_argument('--flavor-col', default='flavors', help='CSV column with flavors (semicolon or JSON array)')
    p.add_argument('--mouthfeel-col', default='mouthfeel', help='CSV column with mouthfeel (semicolon or JSON array)')
    p.add_argument('--name-col', default='name', help='CSV column for wine name')
    p.add_argument('--variety-col', default='variety', help='CSV column for variety')
    p.add_argument('--country-col', default='country', help='CSV column for country')
    p.add_argument('--region1-col', default='region1', help='CSV column for primary region')
    p.add_argument('--region2-col', default='region2', help='CSV column that contains region hierarchy; we will use its LAST token')
    p.add_argument('--age-col', default='age', help='CSV column for age')
    p.add_argument('--price-col', default='price', help='CSV column for real price')
    p.add_argument('--rating-col', default='rating', help='CSV column for real rating')
    p.add_argument('--vocab-json', default=None, help='Optional JSON file with fixed vocab {flavors, mouthfeel, varieties, countries, regions1, regions2, price_bins}')
    p.add_argument('--progress-every', type=int, default=1000, help='Print progress every N rows')
    return p.parse_args()


def parse_tag_list(raw: str) -> List[str]:
    raw = (raw or '').strip()
    if not raw:
        return []
    if raw.startswith('['):
        try:
            arr = json.loads(raw)
            return [str(x) for x in arr]
        except Exception:
            pass
    return [t.strip() for t in raw.split(DEFAULT_TAG_SPLIT) if t.strip()]


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
                variety = row.get(args.variety_col) or None
                country = row.get(args.country_col) or None
                region1 = row.get(args.region1_col) or None
                # region2 source is a hierarchy => take LAST term
                region_h = row.get(args.region2_col) or None
                region2 = last_region_token(region_h)
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

                resp = requests.post(f"{args.api}/embed", json=payload, timeout=30)
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
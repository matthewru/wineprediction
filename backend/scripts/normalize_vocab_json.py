#!/usr/bin/env python3
import json, argparse, unicodedata
from typing import List, Dict, Any

def parse_args():
    p = argparse.ArgumentParser(description='Normalize vocab JSON: lowercase and strip accents')
    p.add_argument('--in', dest='inp', required=True, help='Input vocab JSON path (e.g., global_vocab_v2.json)')
    p.add_argument('--out', dest='out', required=True, help='Output JSON path')
    return p.parse_args()

def norm(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # strip accents
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s.lower()

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_vocab(v: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(v)
    # top-level lists
    for key in [
        'flavor_vocab','mouthfeel_vocab','variety_vocab',
        'price_buckets','rating_buckets','age_buckets'
    ]:
        if key in out and isinstance(out[key], list):
            out[key] = dedupe_keep_order([norm(x) for x in out[key]])
    # geography
    if 'geography_vocab' in out and isinstance(out['geography_vocab'], dict):
        gv = dict(out['geography_vocab'])
        for gk, lst in [
            ('countries', gv.get('countries')),
            ('primary_regions', gv.get('primary_regions')),
            ('secondary_regions', gv.get('secondary_regions')),
        ]:
            if isinstance(lst, list):
                gv[gk] = dedupe_keep_order([norm(x) for x in lst])
        out['geography_vocab'] = gv
    return out

def main():
    args = parse_args()
    with open(args.inp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    normed = normalize_vocab(data)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(normed, f, ensure_ascii=False, indent=2)
    # Print brief sizes
    gv = normed.get('geography_vocab', {})
    print('flavors', len(normed.get('flavor_vocab', [])))
    print('mouthfeel', len(normed.get('mouthfeel_vocab', [])))
    print('varieties', len(normed.get('variety_vocab', [])))
    print('countries', len(gv.get('countries', [])))
    print('primary_regions', len(gv.get('primary_regions', [])))
    print('secondary_regions', len(gv.get('secondary_regions', [])))

if __name__ == '__main__':
    main()

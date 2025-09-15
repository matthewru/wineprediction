#!/usr/bin/env python3
import csv, json, argparse, os
from typing import List, Set

DEFAULT_TAG_SPLIT = ','


def parse_args():
	p = argparse.ArgumentParser(description='Build tags-first global vocab JSON')
	p.add_argument('--csv', required=True, help='Path to wine_clean_tagged.csv')
	p.add_argument('--regions', required=True, help='Path to regions.json')
	p.add_argument('--varieties-json', default=None, help='Optional path to top_27_grape_varieties.json')
	p.add_argument('--out', default='backend/data/global_vocab_v2.json', help='Output vocab JSON')
	p.add_argument('--flavor-col', default='flavor_tags', help='CSV column with flavors (comma-separated)')
	p.add_argument('--mouthfeel-col', default='mouthfeel_tags', help='CSV column with mouthfeel (comma-separated)')
	return p.parse_args()


def split_tags(raw: str) -> List[str]:
	raw = (raw or '').strip()
	if not raw:
		return []
	if raw.startswith('[') and raw.endswith(']'):
		# if JSON-like arrays slipped in
		try:
			arr = json.loads(raw)
			return [str(x).strip() for x in arr if str(x).strip()]
		except Exception:
			pass
	return [t.strip() for t in raw.split(DEFAULT_TAG_SPLIT) if t.strip()]


def main():
	args = parse_args()
	flavors: Set[str] = set()
	mouth: Set[str] = set()

	with open(args.csv, 'r', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			for t in split_tags(row.get(args.flavor_col) or ''):
				flavors.add(t)
			for t in split_tags(row.get(args.mouthfeel_col) or ''):
				mouth.add(t)

	with open(args.regions, 'r', encoding='utf-8') as rf:
		regions = json.load(rf)  # { country: { primary: [secondary] } }
	countries: Set[str] = set()
	primary: Set[str] = set()
	secondary: Set[str] = set()
	for country, primaries in regions.items():
		countries.add(country)
		if isinstance(primaries, dict):
			for pri, secs in primaries.items():
				primary.add(pri)
				if isinstance(secs, list):
					for s in secs:
						secondary.add(s)

	varieties: List[str] = []
	if args.varieties_json and os.path.exists(args.varieties_json):
		with open(args.varieties_json, 'r', encoding='utf-8') as vf:
			vlist = json.load(vf)
			# accept either a list of strings or objects with name field
			if isinstance(vlist, list):
				for v in vlist:
					if isinstance(v, str):
						varieties.append(v)
					elif isinstance(v, dict) and 'name' in v:
						varieties.append(str(v['name']))
	else:
		# fallback: empty (will be provided at runtime)
		varieties = []

	# Buckets: simple, can be tuned later
	price_buckets = ["<15","15-30","30-60","60+","unknown"]
	rating_buckets = ["<80","80-85","85-90","90-95","95+"]
	age_buckets = ["0-1","2-3","4-6","7-10","11+"]

	vocab = {
		"flavor_vocab": sorted(flavors),
		"mouthfeel_vocab": sorted(mouth),
		"geography_vocab": {
			"countries": sorted(countries),
			"primary_regions": sorted(primary),
			"secondary_regions": sorted(secondary),
		},
		"variety_vocab": varieties,
		"price_buckets": price_buckets,
		"rating_buckets": rating_buckets,
		"age_buckets": age_buckets,
	}

	with open(args.out, 'w', encoding='utf-8') as out:
		json.dump(vocab, out, ensure_ascii=False, indent=2)
	print(f"Wrote {args.out}: flavors={len(vocab['flavor_vocab'])}, mouthfeel={len(vocab['mouthfeel_vocab'])}, countries={len(vocab['geography_vocab']['countries'])}, primary={len(vocab['geography_vocab']['primary_regions'])}, secondary={len(vocab['geography_vocab']['secondary_regions'])}, varieties={len(varieties)}")

if __name__ == '__main__':
	main()

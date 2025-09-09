#!/usr/bin/env python3
import csv, json, argparse
from typing import List, Set

DEFAULT_TAG_SPLIT = ';'


def parse_args():
	p = argparse.ArgumentParser(description='Build vocab JSON from wine_clean_tagged CSV')
	p.add_argument('--csv', required=True, help='Path to wine_clean_tagged.csv')
	p.add_argument('--out', default='backend/data/global_vocab.json', help='Output vocab JSON')
	p.add_argument('--flavor-col', default='flavors', help='CSV column with flavors (semicolon or JSON array)')
	p.add_argument('--mouthfeel-col', default='mouthfeel', help='CSV column with mouthfeel (semicolon or JSON array)')
	p.add_argument('--variety-col', default='variety', help='CSV column for variety')
	p.add_argument('--country-col', default='country', help='CSV column for country')
	p.add_argument('--region1-col', default='region1', help='CSV column for primary region')
	p.add_argument('--region2-col', default='region2', help='CSV column for secondary region')
	return p.parse_args()


def parse_tag_list(raw: str) -> List[str]:
	raw = (raw or '').strip()
	if not raw:
		return []
	if raw.startswith('['):
		try:
			arr = json.loads(raw)
			return [str(x).strip() for x in arr if str(x).strip()]
		except Exception:
			pass
	return [t.strip() for t in raw.split(DEFAULT_TAG_SPLIT) if t.strip()]


def main():
	args = parse_args()
	flavors: Set[str] = set()
	mouthfeel: Set[str] = set()
	varieties: Set[str] = set()
	countries: Set[str] = set()
	regions1: Set[str] = set()
	regions2: Set[str] = set()

	with open(args.csv, 'r', newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			for t in parse_tag_list(row.get(args.flavor_col) or ''):
				flavors.add(t)
			for t in parse_tag_list(row.get(args.mouthfeel_col) or ''):
				mouthfeel.add(t)
			v = (row.get(args.variety_col) or '').strip()
			if v:
				varieties.add(v)
			c = (row.get(args.country_col) or '').strip()
			if c:
				countries.add(c)
			r1 = (row.get(args.region1_col) or '').strip()
			if r1:
				regions1.add(r1)
			r2 = (row.get(args.region2_col) or '').strip()
			if r2:
				regions2.add(r2)

	vocab = {
		"flavors": sorted(flavors),
		"mouthfeel": sorted(mouthfeel),
		"varieties": sorted(varieties) + (["Other"] if "Other" not in varieties else []),
		"countries": sorted(countries) + (["Other"] if "Other" not in countries else []),
		"regions1": sorted(regions1),
		"regions2": sorted(regions2),
		"price_bins": ["<15","15-30","30-60","60+","unknown"],
	}

	with open(args.out, 'w', encoding='utf-8') as out:
		json.dump(vocab, out, ensure_ascii=False, indent=2)
	print(f"Wrote vocab to {args.out} with sizes: flavors={len(vocab['flavors'])}, mouthfeel={len(vocab['mouthfeel'])}, varieties={len(vocab['varieties'])}, countries={len(vocab['countries'])}, regions1={len(vocab['regions1'])}, regions2={len(vocab['regions2'])}")


if __name__ == '__main__':
	main() 
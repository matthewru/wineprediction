import json
import re
from pathlib import Path

# Paths
SIGNIFICANT_TERMS_PATH = Path("backend/data/significant_terms.json")
OUTPUT_PATH = Path("backend/data/term_aliases.json")

# Heuristic roots that usually describe mouthfeel / structure / texture
MOUTHFEEL_ROOTS = [
    # Acidity / pH
    "acid",
    # Sweetness / dryness
    "dry", "sweet", "sugar", "sugary",
    # Tannin / grip
    "tannin", "astringent", "grip", "grippy", "chewy",
    # Weight / body
    "full", "light", "medium", "weight", "heavy", "thick", "thin", "dense", "plush", "lush", "plump",
    # Texture descriptors
    "smooth", "silk", "velvet", "cream", "creamy", "chalk", "chalky", "grain", "textur", "rough", "crisp", "slick",
    # Structural / intensity adjectives
    "fresh", "juicy", "complex", "structure", "layer", "depth", "balanced", "balance", "power", "bold", "rich",
    "elegant", "taut", "tight", "lean", "broad", "vibrant", "zest", "zesty", "brisk", "miner", "stone", "stony", "slate", "steely",
    "concentrat", "intens", "long", "linger", "lasting", "mouth", "finish"
]

# Terms to DROP (non-sensory): grape varieties & place names
EXCLUDE_TERMS = {
    # Grapes / varieties
    "cabernet", "cab", "merlot", "pinot", "chardonnay", "sauvignon", "sangiovese", "syrah",
    "zinfandel", "riesling", "malbec", "grenache", "shiraz", "tempranillo", "viognier",
    "mourvèdre", "mourvedre", "verdot", "petit", "petite", "sirah", "carmenère", "carmenere",
    "roussanne", "touriga", "nero", "gris", "blanc",

    # Geography / appellations / places
    "napa", "california", "champagne", "rhône", "rhone",
    "southern", "italy",
}

# Simple helper to decide category

def is_mouthfeel(term: str) -> bool:
    """Return True if term is classified as mouthfeel/structure/texture based on heuristics."""
    lower = term.lower()
    return any(root in lower for root in MOUTHFEEL_ROOTS)


def generate_aliases(term: str) -> list[str]:
    """Generate basic alias list for a term using naive morphological variants."""
    aliases = {term}
    # Pluralisation heuristics
    if term.endswith("y") and len(term) > 2:
        aliases.add(term[:-1] + "ies")
    elif term.endswith("ies") and len(term) > 3:
        aliases.add(term[:-3] + "y")
    # Simple plural rule (add/remove trailing s)
    if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        aliases.add(term[:-1])  # singular
    else:
        aliases.add(term + "s")  # plural

    # Descriptive adjective variations
    # e.g. fruit -> fruity, spice -> spicy, smoke -> smoky
    if not term.endswith("y") and len(term) > 2 and term[-1] not in {"y", "e"}:
        aliases.add(term + "y")
    # Add "ly" adverb form (e.g. crisp -> crisply)
    if not term.endswith("ly") and len(term) > 2:
        aliases.add(term + "ly")
    # Past tense / past participle (e.g. toast -> toasted, age -> aged)
    if not term.endswith("ed") and len(term) > 2:
        if term.endswith("e"):
            aliases.add(term + "d")
        else:
            aliases.add(term + "ed")
    # Present participle (e.g. smoke -> smoking) may not be desired but include for completeness
    if not term.endswith("ing") and len(term) > 3:
        aliases.add(term + "ing")

    # Hyphen / space variations (e.g. full-bodied vs full bodied)
    if "-" in term:
        aliases.add(term.replace("-", " "))
    if " " in term:
        aliases.add(term.replace(" ", "-"))

    return sorted(aliases)


def main():
    if not SIGNIFICANT_TERMS_PATH.exists():
        raise FileNotFoundError(f"Cannot locate {SIGNIFICANT_TERMS_PATH}")

    with SIGNIFICANT_TERMS_PATH.open() as f:
        terms = json.load(f)

    mouthfeel_terms: dict[str, list[str]] = {}
    flavor_terms: dict[str, list[str]] = {}

    for term in terms:
        term_clean = term.strip().lower()
        if not term_clean:
            continue

        # Exclude non-sensory grape/geography terms
        if term_clean in EXCLUDE_TERMS:
            continue

        alias_list = generate_aliases(term_clean)
        if is_mouthfeel(term_clean):
            mouthfeel_terms[term_clean] = alias_list
        else:
            flavor_terms[term_clean] = alias_list

    output = {
        "mouthfeel": dict(sorted(mouthfeel_terms.items())),
        "flavor": dict(sorted(flavor_terms.items())),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"✅ Wrote term aliases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 
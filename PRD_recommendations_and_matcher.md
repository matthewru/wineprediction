## Feature PRD (Informal): For You Recommendations + Real‑Wine Matcher

### Overview
Add two discovery features:
- For You: personalized recommendations sourced from the global bottle pool using content‑based similarity to the user’s cellar.
- Real‑Wine Matcher: find nearest real wines from the original training dataset for a user’s custom wine; show common tags and real price.

### Goals / Success
- For You: Users see at least 10 relevant items; save‑through rate > 10% in early test.
- Matcher: Users can view a top match in < 1s p95; info includes common tags and price.

---

## 1) For You Recommendations

### Scope
- Compute vector embeddings for bottles and users.
- Serve top‑N public bottles via cosine similarity against a user profile embedding.
- UI: new `For You` tab with list of recommendations; supports Save to Cellar.

### Data Model
- Bottle embedding (stored on public `db.bottles`):
  - Concatenate normalized vectors:
    - flavors: fixed index, confidence per flavor (0..1)
    - mouthfeel: fixed index, confidence (0..1)
    - optional features (weights TBD): variety one‑hot, country one‑hot, age bucket, predicted rating, price bucket
  - L2 normalize.
- User profile embedding (stored on `db.users.profile_embedding`):
  - Running EMA over saved bottles (private or public): `profile = 0.9*profile + 0.1*bottle_embedding`.

### Backend Requirements
- New utility: `build_embedding(bottle_doc) -> List[float]` (shared for both features).
- On insert public bottle (already done in POST /cellar when `public: true`):
  - compute embedding, store on the public bottle.
- On any save (private or public):
  - update `users.profile_embedding` (EMA) and persist embedding version.
- New endpoint: `GET /recommendations?limit=20`
  - Auth required.
  - Input: optional `exclude_mine=true` (default), `seed` for A/B.
  - Process: fetch `profile_embedding`; score against cached list of `(bottle_id, embedding)` from public pool; compute cosine, sort, apply small diversity penalty on same variety/region; return top N docs + scores.
- Optional infra:
  - If public set grows large (>100k), use FAISS (IndexFlatIP/HNSW) or MongoDB Atlas Vector Search.

### Frontend Requirements
- Screen: `app/for-you.tsx`
  - Fetch `/recommendations` on focus.
  - Render cards like Global tab; allow Save Private/Public (reuse Results save flow).
  - Empty state: prompt to add wines first (no profile).

### Metrics / Telemetry (future)
- Impression, tap, save actions; simple local logging or backend endpoint.

### Timeline (rough)
- Day 1–2: Embedding function + schema changes; compute on save/public insert.
- Day 3: `/recommendations` endpoint (naïve NumPy scoring + in‑memory cache).
- Day 4: `for-you` screen + wiring; QA on device.
- Day 5: Cleanup (diversity penalty, basic analytics).

---

## 2) Real‑Wine Matcher (Catalog NN)

### Scope
- Precompute embeddings for the original training dataset ("catalog").
- Endpoint to match a custom wine (or any bottle) to top‑K real wines.
- UI: button from Results (and details) to show top match; display common tags and real price.

### Data Model
- New collection `catalog_wines`:
  - `{ _id, name, producer, vintage, country, region, price, tags: {flavors[], mouthfeel[]}, embedding: float32[], embedding_version }`
- Precomputation script generates embeddings for all rows and writes to Mongo.

### Backend Requirements
- Script: `backend/scripts/build_catalog_embeddings.py`
  - Loads training data; if needed, predicts tags using existing models; builds embeddings; writes to `catalog_wines`.
- In‑memory index on startup (small–medium) or FAISS/Vector Search if large.
- New endpoint: `POST /match-real`
  - Body: `{ wine?: bottleDoc, embedding?: number[], top_k?: number }`
  - Build/normalize embedding if not provided.
  - Query top‑K; compute intersections:
    - `common.flavors`: overlap of top‑5 flavor strings
    - `common.mouthfeel`: overlap of top‑5 mouthfeel strings
  - Return: `{ matches: [{ id, name, region, price, score, common }], generated_from: { ... } }`

### Frontend Requirements
- Results screen: add "Find real‑world match" CTA.
- Match modal/screen:
  - Show top match with: name / producer / vintage, real price, common tags (badges), similarity score.
  - List remaining matches (up to 5).

### Timeline (rough)
- Day 1: Precompute script + schema for `catalog_wines`.
- Day 2: Build in‑memory index; implement `/match-real`.
- Day 3: Results UI CTA + details view for match; QA.
- Day 4: Performance polish; optional FAISS/Atlas Vector Search; caching.

---

## Technical Notes / Weights (initial)
- `embedding = [flavor(1.0), mouthfeel(0.7), variety(0.5), country(0.3), age_bucket(0.2), rating(0.2), price_bucket(0.2)]` → normalize.
- Keep `embedding_version` on both bottles and users; bump when changing layout/weights.
- Cold start: If no profile, use last generated wine’s embedding or popular recent public.

## Risks & Mitigations
- Drift between custom generation and catalog tags → always build both embeddings via the same function and vocab.
- Performance with large public pool → switch to FAISS/Atlas vector index.
- Data quality of training set → filter or cap per‑producer duplicates.

## Open Questions
- Minimum N of saves before For You unlocks?
- Diversity policy strength (don’t recommend 5 of the same variety)?
- Do we show external buy links/prices for matched real wines? 
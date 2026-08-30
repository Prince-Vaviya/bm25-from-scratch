# Understanding Search & BM25 from Scratch 🔍

A hands-on implementation to understand the inner working of information retrieval, lexical indexing, **Okapi BM25 ranking** from the ground up.

---

## 🎯 Goals

The purpose of this project is to simulate and understand how search engines work under the hood by building each foundational layer step-by-step:

1. **Text Normalization & Tokenization**: Breaking down raw documents and queries into standardized lexical tokens.
2. **Positional Inverted Indexing**: Constructing an inverted index with positional tracking and document length metadata.
3. **Candidate Document Retrieval**: Fast pruning and candidate selection using posting list lookups.
4. **Okapi BM25 Ranking**: Implementing probabilistic relevance scoring with term frequency saturation ($k_1$) and document length normalization ($b$).
5. **Term & Query Proximity**: Experimenting with sliding-window minimum distance heuristics to boost documents where query terms appear close together.

---

## 🧩 Architecture & Pipeline Flow

```
[ Raw Document Corpus / User Query ]
                 │
                 ▼
       ┌──────────────────┐
       │    Tokenizer     │  (Text Normalization & Regex Extraction)
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │  Inverted Index  │  (Maps Terms ➔ {DocID: Frequency, Positions})
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │Candidate Search  │  (Set-union pruning of posting lists)
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │   BM25 Ranker    │  (IDF + Term Saturation + Length Norm + Proximity)
       └─────────┬────────┘
                 │
                 ▼
     [ Ranked Search Results ]
```

---

## 🔬 Core Components

### 1. Tokenizer (`src/tokenizer.py`)
- Standardizes text by lowercasing and extracting alphanumeric sequences (`[a-z0-9]+`).
- Strips punctuation and whitespace to create uniform lexical units.

### 2. Positional Inverted Index (`src/index.py`)
- Maps each unique term to the documents where it appears.
- Tracks **term frequency** and exact **token position offsets** (`[0, 1, 2, ...]`).
- Maintains corpus statistics including individual document lengths ($dl$) and corpus average document length ($avgdl$).

### 3. Candidate Retrieval (`src/search.py`)
- Prunes the search space by fetching posting lists for all query terms.
- Performs a set union to gather all matching document IDs before ranking.

### 4. Okapi BM25 Ranking (`src/ranking.py`)
Implements the classic Okapi BM25 scoring formula:

$$\text{IDF}(t) = \ln\left( \frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5} + 1 \right)$$

$$\text{TF}_{\text{norm}} = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot \left((1 - b) + b \cdot \frac{dl}{avgdl}\right)}$$

$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \text{TF}_{\text{norm}}$$

- **$k_1$ (default: `1.2`)**: Controls term frequency saturation (diminishing returns for repeated words).
- **$b$ (default: `0.75`)**: Controls document length penalization.

### 5. Term & Query Proximity Scoring (`src/ranking.py`)
- **Pairwise Proximity**: Computes minimum absolute distance between two terms in a document and scores with $\frac{1}{1 + \text{distance}}$.
- **Multi-term Query Proximity**: Uses a sliding window algorithm across positional postings to find the smallest span containing all query terms, scoring with $\frac{1}{\text{window\_size}}$.

---

## 🧪 Experiments & Verification

The `tests/` directory contains targeted simulation scripts to test each unit independently:

| Script | Purpose |
|---|---|
| `python3 -m tests.test_tokenizer` | Tests text tokenization and regex extraction |
| `python3 -m tests.test_index` | Verifies positional inverted index construction |
| `python3 -m tests.test_search` | Verifies candidate document retrieval |
| `python3 -m tests.test_ranking` | Runs BM25 ranking simulations against sample queries |
| `python3 -m tests.test_proximity` | Tests pairwise term proximity calculations |
| `python3 -m tests.test_query_proximity` | Tests multi-term sliding window proximity |
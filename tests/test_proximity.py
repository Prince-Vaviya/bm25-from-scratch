from src.tokenizer import tokenize
from src.index import InvertedIndex
from src.search import retrieve_candidates
from src.ranking import BM25Ranker


index = InvertedIndex()

# Build index
for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as content:
        text = content.read()

    index.add_document(i, tokenize(text))


ranker = BM25Ranker(index)


queries = [
    "solar energy",
    "human brain",
    "network protocol"
]


for query in queries:

    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print(f"{'=' * 60}")

    candidates = retrieve_candidates(query, index)
    terms = list(set(tokenize(query)))

    for document_id in candidates:

        bm25 = ranker.score(document_id, query)

        proximity = 0

        if len(terms) >= 2:
            distance = ranker.term_proximity(
                document_id,
                terms[0],
                terms[1]
            )

            if distance != float("inf"):
                proximity = 1 / (1 + distance)

        combined = bm25 + (0.2 * proximity)

        print(
            f"D{document_id} | "
            f"BM25: {bm25:.4f} | "
            f"Proximity: {proximity:.4f} | "
            f"Combined: {combined:.4f}"
        )
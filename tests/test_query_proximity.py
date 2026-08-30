from src.tokenizer import tokenize
from src.index import InvertedIndex
from src.ranking import BM25Ranker


index = InvertedIndex()

for i in range(1, 7):

    with open(f"data/{i}.txt", "r") as content:
        text = content.read()

    index.add_document(i, tokenize(text))


ranker = BM25Ranker(index)


queries = [
    "solar energy",
    "human brain",
    "network protocol",
]


for query in queries:

    print(f"\nQUERY: {query}")

    for document_id in range(1, 7):

        proximity = ranker.query_proximity(
            document_id,
            query
        )

        if proximity > 0:

            print(
                f"D{document_id} → "
                f"{proximity:.4f}"
            )
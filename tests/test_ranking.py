from src.ranking import BM25Ranker
from src.tokenizer import tokenize
from src.index import InvertedIndex
from src.search import retrieve_candidates

index = InvertedIndex()

for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as content:
        file_contents = content.read()

    index.add_document(i, tokenize(file_contents))

ranker = BM25Ranker(index)

distance = ranker.term_proximity(3, "solar", "energy")

print(distance)
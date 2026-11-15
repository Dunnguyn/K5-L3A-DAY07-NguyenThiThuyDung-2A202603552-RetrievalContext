"""Day 7 original-query retrieval experiments. Run from the repository root."""
import argparse
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

from baseline import TfidfIndex, evaluate, tokenize, validate

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
OUT = Path(os.environ.get("DAY7_RESULTS_DIR", "report/results"))
CONFIGS = {"tfidf": ("tfidf", 800), "bm25": ("bm25", 800),
           "dense": ("dense", 800), "rrf": ("rrf", 800),
           "rrf-wide": ("rrf", 3200), "bm25-wide": ("bm25", 3200)}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


class BM25:
    def __init__(self, chunks):
        self.rows = [(c, Counter(tokenize(c["text"]))) for c in chunks]
        df = Counter(term for _, counts in self.rows for term in counts)
        self.idf = {t: math.log1p((len(chunks)-n+0.5)/(n+0.5)) for t,n in df.items()}
        self.avg = statistics.mean(counts.total() for _, counts in self.rows)

    def search(self, question, top_k):
        terms = set(tokenize(question))
        results = []
        for chunk, counts in self.rows:
            norm = 1.5 * (0.25 + 0.75 * counts.total() / self.avg)
            score = sum(self.idf[t] * counts[t] * 2.5 / (counts[t] + norm)
                        for t in terms if counts[t])
            if score:
                results.append(dict(chunk, score=score, strategy="bm25"))
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class Dense:
    def __init__(self, chunks):
        import chromadb
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        self.model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
        self.docs = {c["id"]: c for c in chunks}
        path = os.environ.get("DAY6_CHROMA_PATH")
        if path:
            if not (Path(path) / "chroma.sqlite3").is_file():
                raise ValueError("Missing Day 6 Chroma database")
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_collection("day06_sensor_b", embedding_function=None)
            metadata = self.collection.metadata or {}
            if (metadata.get("embedding_model"), metadata.get("dimensions"), metadata.get("hnsw:space")) != (MODEL, 384, "cosine"):
                raise ValueError("Incompatible Day 6 collection settings")
            snapshot = self.collection.get(include=["documents","metadatas"])
            if set(snapshot["ids"]) != set(self.docs):
                raise ValueError("Day 6 IDs differ from benchmark corpus")
            for key, doc, meta in zip(snapshot["ids"], snapshot["documents"], snapshot["metadatas"]):
                chunk = self.docs[key]
                if (meta["document_id"], meta["source"]) != (chunk["source_id"], chunk["source"]):
                    raise ValueError(f"Day 6 source mismatch: {key}")
                self.docs[key] = dict(chunk, text=doc, page=meta["page"])
        else:
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.create_collection("sensor_day7", metadata={"hnsw:space":"cosine"})
            self.collection.add(ids=list(self.docs), embeddings=self.embed([c["text"] for c in chunks]))

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def search(self, question, top_k):
        rows = self.collection.query(query_embeddings=self.embed([question]), n_results=min(top_k,len(self.docs)))
        return [dict(self.docs[i], score=1-d, strategy="dense")
                for i,d in zip(rows["ids"][0], rows["distances"][0])]


class Fusion:
    def __init__(self, lexical, dense):
        self.indices = (lexical, dense)

    def search(self, question, top_k):
        scores, docs = Counter(), {}
        for index in self.indices:
            for rank, c in enumerate(index.search(question, 10), 1):
                scores[c["id"]] += 1/(60+rank)
                docs[c["id"]] = c
        return [dict(docs[i], score=scores[i], strategy="rrf")
                for i in sorted(scores, key=lambda i: (-scores[i], i))[:top_k]]


def indices(chunks, methods):
    result = {"tfidf": TfidfIndex(chunks), "bm25": BM25(chunks)}
    if {"dense","rrf"} & set(methods):
        result["dense"] = Dense(chunks)
        result["rrf"] = Fusion(result["bm25"], result["dense"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev","test"])
    split = parser.parse_args().split
    names = list(CONFIGS) if split == "dev" else list(dict.fromkeys(["tfidf", load(OUT/"selection.json")["config"]]))
    chunks = load("data/chunks.json")
    questions = [q for q in load("data/questions.json") if q["split"] == split]
    validate(chunks, questions)
    searchers = indices(chunks, [CONFIGS[n][0] for n in names])
    summaries = {}
    for name in names:
        method, budget = CONFIGS[name]
        traces = evaluate(searchers[method], questions, budget)
        for t in traces:
            t["strategy"] = method
        save(f"{split}-{name}.json", traces)
        summaries[name] = {k: sum(t[k] for t in traces) for k in ["hit1","hit3","context_hit"]}
        summaries[name].update(n=len(traces), median_ms=statistics.median(t["elapsed_ms"] for t in traces))
    save(f"{split}-summary.json", summaries)
    if split == "dev":
        criteria = ["context_hit","hit3","hit1"]
        best = max(names, key=lambda n: (*[summaries[n][k] for k in criteria], -summaries[n]["median_ms"]))
        save("selection.json", {"config":best, "criteria":criteria + ["lowest median_ms"]})
    print(summaries)


if __name__ == "__main__":
    main()

"""Compare Day 6 persisted rankings with the submitted Day 7 traces."""
import os

from experiment import CONFIGS, OUT, indices, load, save


def main():
    if not os.environ.get("DAY6_CHROMA_PATH"):
        raise SystemExit("Set DAY6_CHROMA_PATH to a copy of the Day 6 Chroma directory")
    searchers = indices(load("data/chunks.json"), ["rrf"])
    comparisons = []
    for split, names in [("dev", ["dense", "rrf", "rrf-wide"]), ("test", ["rrf-wide"])]:
        for name in names:
            method, _ = CONFIGS[name]
            traces = load(OUT / f"{split}-{name}.json")
            for trace in traces:
                results = searchers[method].search(trace["question"], 3)
                ids = [r["id"] for r in results]
                comparisons.append({"split": split, "config": name, "query_id": trace["query_id"],
                                    "top3": ids, "same_top3": ids == [r["id"] for r in trace["results"]]})
    save("day6-store-check.json", comparisons)
    print(sum(row["same_top3"] for row in comparisons), "/", len(comparisons))


if __name__ == "__main__":
    main()

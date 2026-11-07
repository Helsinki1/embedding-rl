"""Query-only decomposition handling and set union retrieval."""


def components_for(query, decomposition):
    if decomposition["query"] != query:
        raise ValueError("Decomposition cache query mismatch")
    components = decomposition.get("components", [])
    valid = (decomposition.get("keep") is True and isinstance(components, list)
             and len(components) in (2, 3)
             and all(isinstance(c, str) and c.strip() for c in components))
    if valid:
        normalized = [" ".join(c.split()).casefold() for c in components]
        valid = len(set(normalized)) == len(components)
    return (components, False) if valid else ([query], True)


def flatten_queries(rows, decompositions):
    flat, offsets, fallbacks = [], [], []
    for row in rows:
        components, fallback = components_for(row["query"], decompositions[row["query"]])
        offsets.append((len(flat), len(flat) + len(components)))
        flat.extend({"query": c} for c in components)
        if fallback:
            fallbacks.append(row["id"])
    return flat, offsets, fallbacks


def union_predictions(rows, offsets, rankings, titles, k):
    if len(rows) != len(offsets) or (offsets and offsets[-1][1] != len(rankings)):
        raise ValueError("Component rankings are not aligned to query rows")
    predictions = []
    for row, (start, end) in zip(rows, offsets):
        docs = dict.fromkeys(titles[index] for ranking in rankings[start:end] for index in ranking[:k])
        predictions.append({"id": row["id"], "query": row["query"], "docs": list(docs)})
    return predictions

"""Score the same final, deduplicated result set against union and component gold."""
from collections import defaultdict


def set_metrics(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    hits = len(gold & predicted)
    return {"precision": hits / len(predicted) if predicted else 0.0,
            "recall": hits / len(gold) if gold else None,
            "f1": 2 * hits / (len(gold) + len(predicted)) if gold else None,
            "hits": hits, "gold_count": len(gold), "predicted_count": len(predicted)}


def average(rows, keys=("precision", "recall", "f1", "predicted_count", "gold_count")):
    return {k: sum(r[k] for r in rows if r[k] is not None) / sum(r[k] is not None for r in rows)
            if any(r[k] is not None for r in rows) else None for k in keys}


def evaluate(rows, predictions):
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Predictions must cover each evaluation query exactly once")
    details = []
    by_source, by_arity = defaultdict(list), defaultdict(list)
    for row in rows:
        predicted = by_id[row["id"]]["docs"]
        union = set_metrics(row["docs"], predicted)
        components = [{"component": c, **set_metrics(g, predicted)}
                      for c, g in zip(row["components"], row["component_docs"])]
        scored = [c for c in components if c["gold_count"]]
        component = average(scored, ("precision", "recall", "f1"))
        detail = {"id": row["id"], "source": row["source"], **union,
                  "component_macro": component, "components": components,
                  "component_coverage": sum(c["hits"] > 0 for c in scored) / len(scored)}
        details.append(detail)
        by_source[row["source"]].append(detail)
        by_arity[str(len(components))].append(detail)

    def summarize(items):
        return {"queries": len(items), "union": average(items),
                "component_macro": average([r["component_macro"] for r in items], ("precision", "recall", "f1")),
                "component_coverage": sum(r["component_coverage"] for r in items) / len(items),
                "unscorable_components": sum(c["gold_count"] == 0 for r in items for c in r["components"])}

    return {"overall": summarize(details), "by_source": {k: summarize(v) for k, v in by_source.items()},
            "by_arity": {k: summarize(v) for k, v in by_arity.items()}}, details

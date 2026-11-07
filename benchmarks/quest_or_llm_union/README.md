# Untrained LLM decomposition baseline

This baseline uses the same frozen 3,600/400/1,000 query split and the complete
393,140-document corpus as the shared training experiments. It performs no
training and does not use the training queries.

For each validation/test query, the original query-only prompt asks
`gpt-5.4-mini-2026-03-17` for two or three self-contained OR alternatives. Only
the query string is supplied to the API. Synthetic component annotations, gold
documents, and source metadata are not inputs. Matching original QUEST responses
are reused only when both model and prompt hash match. All other responses are
generated and cached. The input format follows the official [Structured Outputs
guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

Every query remains in the test set. If the LLM rejects the OR interpretation,
refuses, or returns an invalid decomposition, the original whole query is used.
API transport failures stop the run after SDK retries; the saved cache supports
resuming without repeating successful calls.

Each returned component is embedded with unchanged Qwen3-Embedding-8B weights,
the existing query instruction, last-token pooling, and 4096-dimensional
normalization. The implementation uses the same zero-initialized LoRA wrapper
as the frozen single-query baseline, freezes all parameters, and makes no
optimizer updates. Each component retrieves its top k documents from the full
corpus by cosine similarity. Results are unioned and deduplicated by document
ID; the union is not truncated back to k.

Choose k from 1–20 by validation macro F1, preferring smaller k on ties. Save
that selection before evaluating the 1,000 test queries. Precision, recall,
and F1 are macro averages of per-query set metrics. The synthetic labels are
supplied positive unions, not exhaustive relevance judgments.

Run decomposition locally, using `OPENAI_KEY` or `OPENAI_API_KEY` from the local
`.env`; secrets are never copied into code or result bundles:

```bash
python -m benchmarks.quest_or_llm_union.decompose --output /path/to/decomposition
```

Run retrieval on the compute server with the frozen corpus/model paths:

```bash
python -m benchmarks.quest_or_llm_union.evaluate \
  --corpus-dataset /path/to/quest_or_synthetic \
  --work-dir /path/to/quest-or-baseline \
  --embeddings /path/to/synthetic/embeddings \
  --decompositions /path/to/decomposition \
  --output /path/to/new/evaluation
```

The saved manifest, API response IDs/usage, model/prompt hashes, shared split
hash, validation curve, test predictions, and per-query metrics make the result
auditable. Decomposition is capped at 25 minutes and GPU evaluation at 5 minutes
for this run.

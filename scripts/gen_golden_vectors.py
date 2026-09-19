#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "sentence-transformers==3.4.1",
#     "torch==2.6.0",
#     "transformers==4.49.0",
#     "tokenizers==0.21.0",
#     "einops==0.8.1",
# ]
# ///
"""Generate reference embeddings for the golden-vector test fixture.

Generated with Claude. Human-reviewed before merge -- see CONTRIBUTING.md.

This is NOT a test. It runs locally, on demand, and writes a fixture that is
committed to the repo. CI compares against the committed file and never runs
this script -- otherwise every test run would pull torch and download model
weights over the network.

Regenerating is a deliberate act. When a change moves the numbers on purpose
(see issue #33), rerun this and let the diff on the CSV be the evidence for
the release note.

    uv run scripts/gen_golden_vectors.py

Dependencies are pinned in the PEP 723 block above -- uv resolves them into a
throwaway environment and fetches a compatible interpreter, so nothing is
installed into the project and the repo venv (Python 3.14, which torch has no
wheels for) is left alone.

Bump those pins deliberately, never incidentally: a version change there can
move the reference vectors, which is exactly what this fixture exists to catch.

Output: test/golden/golden_vectors.csv
"""

import csv
import json
import pathlib
import sys

from sentence_transformers import SentenceTransformer

OUT = pathlib.Path(__file__).resolve().parent.parent / "test" / "golden" / "golden_vectors.csv"

MODELS = [
    # (fixture name, HF model id, expected dims, trust_remote_code)
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, False),
    ("jina", "jinaai/jina-embeddings-v2-base-en", 768, True),
]

# Deterministic filler for the length-boundary cases. Never randomise: the
# fixture must be byte-reproducible from this script alone.
FILLER = (
    "the quick brown fox jumps over the lazy dog while the barge drifts "
    "downstream past the old mill and the heron waits in the shallows "
).split()


def text_of_n_tokens(tokenizer, n):
    """Build a string that encodes to approximately n tokens.

    Binary-searches on word count rather than growing one word at a time --
    at Jina's 8192-token limit the naive loop re-encodes a growing string
    thousands of times and takes minutes.

    Lands within a token or two of the target, which is all we need: these
    cases exist to straddle the context boundary, not to hit it exactly.
    """
    def n_tokens(word_count):
        words = [FILLER[i % len(FILLER)] for i in range(word_count)]
        return len(tokenizer.encode(" ".join(words))), " ".join(words)

    lo, hi = 1, 8
    while n_tokens(hi)[0] < n:
        lo, hi = hi, hi * 2
        if hi > 1 << 20:
            raise RuntimeError(f"cannot reach {n} tokens")

    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        count, text = n_tokens(mid)
        if count <= n:
            best, lo = text, mid + 1
        else:
            hi = mid - 1
    return best


def build_cases(model, limit):
    """One case per failure mode we are about to touch. See issue #29."""
    tok = model.tokenizer
    return [
        ("short_ascii", "The quick brown fox jumps over the lazy dog."),
        ("accented", "¿Cuál es la capital de Chile? Santiago está en el valle central."),
        ("non_latin", "東京は日本の首都です。"),
        ("empty", ""),
        # Straddles the context limit: #34 truncation
        ("near_limit", text_of_n_tokens(tok, limit - 12)),
        ("over_limit", text_of_n_tokens(tok, limit + 200)),
        # Pair for the batch-composition test: #33. Embedded here in isolation,
        # so these rows are the ground truth that batched inference must match.
        ("batch_short", "Short row."),
        ("batch_long", text_of_n_tokens(tok, 400)),
    ]


def main():
    rows = []
    for name, model_id, dims, remote in MODELS:
        print(f"\n=== {name}  ({model_id}) ===", file=sys.stderr)
        model = SentenceTransformer(model_id, trust_remote_code=remote)

        limit = model.max_seq_length
        print(f"  max_seq_length : {limit}", file=sys.stderr)
        print(f"  modules        : {[type(m).__name__ for m in model]}", file=sys.stderr)
        # Verify the pooling matches what embed_utils.rs does (mean + L2).
        for module in model:
            if type(module).__name__ == "Pooling":
                modes = [k for k, v in module.get_config_dict().items()
                         if k.startswith("pooling_mode") and v]
                print(f"  pooling        : {modes}", file=sys.stderr)

        cases = build_cases(model, limit)
        texts = [t for _, t in cases]
        # normalize_embeddings must stay True: embed_utils.rs L2-normalises
        # (normalize_l2, line ~271). Comparing a normalised vector against an
        # unnormalised one looks like catastrophic failure but is not.
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=1)

        for (case_id, text), vec in zip(cases, vecs):
            assert len(vec) == dims, f"{name}/{case_id}: got {len(vec)} dims, expected {dims}"
            n_tok = len(model.tokenizer.encode(text))
            if any(v != v for v in vec):  # NaN check
                print(f"  !! {case_id}: NaN in reference vector", file=sys.stderr)
            print(f"  {case_id:<12} {n_tok:>5} tokens", file=sys.stderr)
            rows.append({
                "model": name,
                "case_id": case_id,
                "n_tokens": n_tok,
                "text": text,
                "vector": json.dumps([round(float(v), 8) for v in vec]),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "case_id", "n_tokens", "text", "vector"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} reference vectors to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()

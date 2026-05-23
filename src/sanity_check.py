"""
Sanity-check script for the Tokenizer ABC and the WikiTokenizer implementation.

Run from the `src/` directory:

    python sanity_check.py

Expects a pretrained Wikipedia2Vec model under `../models/` (any `.pkl` or
`.bin` file). Pretrained models can be downloaded from the Wikipedia2Vec
project's "Pretrained Embeddings" page.
"""

from __future__ import annotations

import glob
import os
import sys

from src.tokenizers.tokenizer import Token
from src.tokenizers.wiki_tokenizer import WikiTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MISSING_TOKEN = "zzzqqqxxx_not_a_real_token_12345"


def find_model() -> str:
    """Return the path of the first Wikipedia2Vec model found in MODEL_DIR."""
    candidates = sorted(
        glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
        + glob.glob(os.path.join(MODEL_DIR, "*.bin"))
    )
    if not candidates:
        sys.exit(
            f"No Wikipedia2Vec model (.pkl or .bin) found in {MODEL_DIR}. "
            "Download one from the Wikipedia2Vec project's pretrained "
            "embeddings page and drop it there."
        )
    return candidates[0]


def main() -> None:
    model_path = find_model()
    print(f"Loading model: {model_path}")
    tok = WikiTokenizer(model_path)
    print(f"  dim = {tok.dim}")

    # --- get_vector: known word ---
    vec = tok.get_vector("computer")
    assert vec is not None, "expected 'computer' in the word vocabulary"
    assert vec.shape == (tok.dim,), f"unexpected shape {vec.shape}"
    print(f"get_vector('computer') -> shape {vec.shape}")

    # --- get_vector: known entities (celebrity Wikipedia pages) ---
    celebrities = ["Barack Obama", "Taylor Swift", "Leonardo DiCaprio"]
    for name in celebrities:
        v = tok.get_vector(name)
        assert v is not None, f"expected entity '{name}' in vocab"
        assert v.shape == (tok.dim,)
    print(f"get_vector for celebrities OK: {celebrities}")

    # --- get_vector: Token with is_entity selects the vocabulary exactly ---
    word_obama = tok.get_vector(Token("obama", is_entity=False))
    entity_obama = tok.get_vector(Token("Barack Obama", is_entity=True))
    assert word_obama is not None and entity_obama is not None
    # Entity-flagged lookup of a word-only string should miss, and vice versa.
    assert tok.get_vector(Token("Barack Obama", is_entity=False)) is None
    print("get_vector with Token(is_entity=...) OK")

    # --- get_vector: nonsense token returns None ---
    assert tok.get_vector(MISSING_TOKEN) is None
    print("get_vector for nonsense -> None (OK)")

    # --- __contains__ ---
    assert "computer" in tok
    assert "Barack Obama" in tok
    assert MISSING_TOKEN not in tok
    print("__contains__ OK")

    # --- similarity: related people should be more similar than unrelated ones ---
    sim_related = tok.similarity("Barack Obama", "Joe Biden")
    sim_unrelated = tok.similarity("Barack Obama", "banana")
    assert sim_related is not None and sim_unrelated is not None
    print(f"similarity(Obama, Biden)  = {sim_related:+.3f}")
    print(f"similarity(Obama, banana) = {sim_unrelated:+.3f}")
    assert sim_related > sim_unrelated, "Obama/Biden should beat Obama/banana"

    # similarity with a missing token -> None
    assert tok.similarity("Barack Obama", MISSING_TOKEN) is None
    print("similarity missing-token -> None (OK)")

    # --- most_similar: query excluded, length capped, types as expected ---
    neighbors = tok.most_similar("Taylor Swift", k=5)
    print("most_similar('Taylor Swift', k=5):")
    for token, score in neighbors:
        print(f"  {score:+.3f}  {token!r}")
    assert len(neighbors) <= 5
    assert all(token != "Taylor Swift" for token, _ in neighbors)
    assert all(isinstance(t, Token) and isinstance(s, float) for t, s in neighbors)

    # missing query -> []
    assert tok.most_similar(MISSING_TOKEN) == []
    print("most_similar missing-token -> [] (OK)")

    # --- Token disambiguation: word vs. entity with the same surface form ---
    # "obama" exists as a lowercase word; "Barack Obama" as an entity page.
    # They should produce distinct Tokens whose `url` reflects entity-ness.
    word_tok = Token("obama", is_entity=False)
    entity_tok = Token("Barack Obama", is_entity=True)
    assert word_tok != entity_tok
    assert word_tok.url is None
    assert entity_tok.url == "https://en.wikipedia.org/wiki/Barack_Obama"
    print(f"Token disambiguation OK ({entity_tok!r} -> {entity_tok.url})")

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()

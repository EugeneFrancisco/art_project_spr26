"""
Build and visualize a pairwise cosine-similarity table over the artists in
`data/artist_groupings/artists_by_style.xlsx`.

The artists are kept in the order they appear in the sheet (grouped by style),
so style clusters show up as blocks along the diagonal of the heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.tokenizers.tokenizer import Token
from src.tokenizers.wiki_tokenizer import WikiTokenizer

ARTISTS_PATH = "data/artist_groupings/artists_by_style.xlsx"
HEATMAP_PATH = "data/artist_groupings/artist_similarity_heatmap.png"
MATRIX_PATH = "data/artist_groupings/artist_similarity_matrix.npy"
NAMES_PATH = "data/artist_groupings/artist_similarity_names.txt"


def load_artists(path: str = ARTISTS_PATH) -> list[tuple[str, str]]:
    """
    Return `(artist, style)` pairs in the order they appear in the sheet,
    deduplicated on artist (first occurrence wins, so each artist is assigned
    to the first style they were listed under).
    """
    df = pd.read_excel(path)
    df = df.drop_duplicates(subset="Artist", keep="first")
    return list(zip(df["Artist"].tolist(), df["Style"].tolist()))


def build_embedding_matrix(
    artists: list[tuple[str, str]], wiki: WikiTokenizer
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Look up each artist's entity embedding. Returns the (n, d) embedding matrix,
    the matching list of resolved names, and the parallel list of styles.
    Artists missing from the model are skipped (and reported on stdout).
    """
    artist_vec = wiki.get_vector(Token("artist", is_entity=False))
    if artist_vec is None:
        raise RuntimeError("Word 'artist' missing from Wikipedia2Vec vocabulary.")
    artist_dir = artist_vec / np.linalg.norm(artist_vec)

    vectors: list[np.ndarray] = []
    resolved: list[str] = []
    styles: list[str] = []
    missing: list[str] = []
    for name, style in artists:
        vec = wiki.get_vector(Token(name, is_entity=True))
        if vec is None:
            missing.append(name)
            continue
        vec = vec - (vec @ artist_dir) * artist_dir
        vectors.append(vec)
        resolved.append(name)
        styles.append(style)
    if missing:
        print(f"Skipped {len(missing)} artists not in Wikipedia2Vec:")
        for name in missing:
            print(f"  - {name}")
    return np.vstack(vectors), resolved, styles


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of the rows of `embeddings`."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _contiguous_runs(labels: list[str]) -> list[tuple[int, int, str]]:
    """Return `(start, end_exclusive, label)` for each consecutive run in `labels`."""
    runs: list[tuple[int, int, str]] = []
    if not labels:
        return runs
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    runs.append((start, len(labels), labels[start]))
    return runs


def plot_heatmap(
    sim: np.ndarray,
    styles: list[str] | None = None,
    out_path: str = HEATMAP_PATH,
) -> None:
    """
    Render `sim` as an unlabeled square heatmap and save to `out_path`. When
    `styles` is provided, draw a box along the diagonal around each contiguous
    run of artists sharing a style.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sim, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Artist cosine similarity ({sim.shape[0]} x {sim.shape[0]})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if styles is not None:
        for start, end, _ in _contiguous_runs(styles):
            ax.add_patch(Rectangle(
                (start - 0.5, start - 0.5),
                end - start,
                end - start,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            ))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()


def generate_similarity_table(wiki: WikiTokenizer) -> tuple[np.ndarray, list[str]]:
    """
    Run the full pipeline: load artists, look up embeddings, build the cosine
    similarity matrix, persist it (and the matching names) to disk, and render
    the heatmap.

    Returns the (n, n) similarity matrix and the list of resolved artist names.
    """
    artists = load_artists()
    print(f"Loaded {len(artists)} unique artists from {ARTISTS_PATH}")

    embeddings, resolved, styles = build_embedding_matrix(artists, wiki)
    print(f"Resolved {len(resolved)} / {len(artists)} artists to embeddings "
          f"(dim={embeddings.shape[1]})")

    sim = cosine_similarity_matrix(embeddings)
    print(f"Similarity matrix shape: {sim.shape}")

    np.save(MATRIX_PATH, sim)
    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(resolved))
    print(f"Saved matrix to {MATRIX_PATH} and names to {NAMES_PATH}")

    plot_heatmap(sim, styles=styles)
    print(f"Saved heatmap to {HEATMAP_PATH}")

    return sim, resolved

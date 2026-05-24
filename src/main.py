"""
Main script for experimenting.
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer
from src.visualizations.similarity_table import generate_similarity_table

WIKI_PATH = "models/wiki_models/enwiki_20180420_100d.pkl"


def main():
    wiki = WikiTokenizer(WIKI_PATH)
    generate_similarity_table(wiki)


if __name__ == "__main__":
    main()

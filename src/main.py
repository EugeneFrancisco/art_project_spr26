"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer

WIKI_PATH = "models/enwiki_20180420_100d.pkl"

def main():
    wiki = WikiTokenizer(WIKI_PATH)
    


if __name__ == "__main__":
    main()
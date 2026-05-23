"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer

WIKI_PATH = "models/enwiki_20180420_500d.pkl"

def main():
    wiki = WikiTokenizer(WIKI_PATH)
    matches = wiki.most_similar("Mona Lisa", k = 10)
    print(matches)
    


if __name__ == "__main__":
    main()
"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer

WIKI_PATH = "models/enwiki_20180420_100d.pkl"

def main():
    wiki = WikiTokenizer(WIKI_PATH)
    pollock = wiki.get_vector("Gerhard Richter")
    neighbors = wiki.get_neighbors(pollock, 5)
    print(neighbors)



if __name__ == "__main__":
    main()
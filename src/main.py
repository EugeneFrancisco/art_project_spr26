"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer

WIKI_PATH = "models/enwiki_20180420_100d.pkl"

def main():
    wiki = WikiTokenizer(WIKI_PATH)
    # Pollock - Picasso = Abstract Expressionism - Cubism
    # Pollock - Abstract Expressionism + Cubism = _____
    
    query = wiki.analogy(
        positives=["Gerhard Richter", "Jackson Pollock"],
        negatives=["Anselm Kiefer"],
        k=10
    )
    print(query)
    


if __name__ == "__main__":
    main()
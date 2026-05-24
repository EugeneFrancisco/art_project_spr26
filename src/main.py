"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer
from src.data_utils.shi_scraper import ShiScraper

WIKI_PATH = "models/enwiki_20180420_500d.pkl"
SHI_DATA_PATH = "data/shi_data.txt"

def main():
    shi_scraper = ShiScraper(SHI_DATA_PATH)
    wiki = WikiTokenizer(WIKI_PATH)
    names, prices = shi_scraper.get_data_in_wiki(wiki, True)
    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    main()
"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer
from src.data_utils.shi_scraper import ShiScraper
from src.trainer.dnn import DNNVec2Price
from src.trainer.knn import KNNVec2Price

import numpy as np

WIKI_PATH = "models/wiki_models/enwiki_20180420_100d.pkl"
SHI_DATA_PATH = "data/shi_data.txt"

def main():
    shi_scraper = ShiScraper(SHI_DATA_PATH)
    wiki = WikiTokenizer(WIKI_PATH)

    names, prices = shi_scraper.get_data_in_wiki(
        wiki=wiki,
        only_entries=True,
        aggregate=True,
        method="median"
    )

    log_prices = np.log(prices)
    baseline = np.var(log_prices)
    print(f"\nBaseline is {baseline}\n")

    knn_predictor = KNNVec2Price(
        tokenizer=wiki,
        names=names,
        prices=log_prices,
        k=5,
        weighted=True
    )

    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    main()
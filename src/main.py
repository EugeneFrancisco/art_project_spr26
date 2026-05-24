"""
Main script for experimenting
"""
# pylint: skip-file

from src.tokenizers.wiki_tokenizer import WikiTokenizer
from src.data_utils.shi_scraper import ShiScraper
from src.trainer.dnn import DNNVec2Price

import numpy as np

WIKI_PATH = "models/wiki_models/enwiki_20180420_100d.pkl"
SHI_DATA_PATH = "data/shi_data.txt"

def main():
    shi_scraper = ShiScraper(SHI_DATA_PATH)
    wiki = WikiTokenizer(WIKI_PATH)
    names, prices = shi_scraper.get_data_in_wiki(wiki, True)
    log_prices = np.log(prices)
    baseline = np.var(log_prices)
    print(f"\nBaseline is {baseline}\n")

    dnn_predictor = DNNVec2Price(
        tokenizer=wiki,
        names=names,
        prices=log_prices,
        hidden_dims=(512, 256, 128, 64)
    )
    dnn_predictor.train()
    print(dnn_predictor.test())

    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    main()
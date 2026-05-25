"""
Main script for experimenting.
"""
# pylint: skip-file

import numpy as np

from src.tokenizers.wiki_tokenizer import WikiTokenizer
from src.data_utils.shi_scraper import ShiScraper
from src.data_utils.kuhm_scraper import KuhmScraper
from src.trainer.dnn import DNNVec2Price
from src.visualizations.similarity_table import generate_similarity_table
from src.visualizations.utils import plot_histogram

WIKI_PATH = "models/wiki_models/enwiki_20180420_500d.pkl"
SHI_PATH = "data/shi_data/shi_data.txt"
KUHM_PATH = "data/kaggle_art_price_dataset/artDataset.csv"


def main():
    wiki = WikiTokenizer(WIKI_PATH)
    # shi_scraper = ShiScraper(SHI_PATH)
    # names, prices = shi_scraper.get_data_in_wiki(
    #     wiki=wiki,
    #     only_entries=True,
    #     aggregate=True
    # )
    # log_prices = np.log(prices)

    # plot_histogram(log_prices)

    kuhm_scraper = KuhmScraper(KUHM_PATH)
    names, prices = kuhm_scraper.get_data_in_wiki(
        wiki=wiki,
        only_entries=True,
        aggregate=True
    )
    log_prices = np.log(prices)
    plot_histogram(log_prices)


    dnn = DNNVec2Price(
        tokenizer=wiki,
        names=names,
        prices=log_prices,
        hidden_dims=(256, 128, 64),
        device="mps"
    )

    dnn.train()
    print(f"DNN test: {dnn.test()}")
    print(f"baseline test: {dnn.test_baseline()}")
    
    # target = wiki.analogy(
    #     positives=["Jackson Pollock", "Cubism"],
    #     negatives=["Abstract Expressionism"]
    # )
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()

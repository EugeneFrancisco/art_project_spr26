"""
Scraper for the Kaggle art price dataset (artDataset.csv) by Kuhm.
"""
import csv
from typing import Optional, Callable

import numpy as np

from src.data_utils.data_wrapper import AuctionScraper


PATH = "data/kaggle_art_price_dataset/artDataset.csv"


class KuhmScraper(AuctionScraper):
    """
    Parses artDataset.csv and emits one (artist, price) pair per sale.

    Prices are formatted like "1.500 USD" using a period as the thousands
    separator (European style), so "1.500 USD" means $1,500 and "800 USD"
    means $800. Rows with missing/unparseable price or artist are skipped.
    """

    def __init__(self, path: str = PATH):
        self.path = path

    def get_data(
            self,
            acceptance_function: Optional[Callable[[str], bool]] = None
        ) -> tuple[list, np.ndarray]:
        names: list[str] = []
        prices: list[float] = []

        with open(self.path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                artist = (row.get("artist") or "").strip()
                price_str = (row.get("price") or "").strip()
                if not artist or not price_str:
                    continue
                if acceptance_function is not None and not acceptance_function(artist):
                    continue
                try:
                    amount = price_str.rsplit(" ", 1)[0].replace(".", "")
                    price = float(amount)
                except ValueError:
                    continue
                if price <= 0:
                    continue
                names.append(artist)
                prices.append(price)

        return names, np.array(prices)

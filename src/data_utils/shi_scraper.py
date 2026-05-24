"""
Scraper for the shi_data.txt auction history file.
"""
from typing import Optional, Callable

import numpy as np

from src.data_utils.data_wrapper import AuctionScraper


PATH = "data/shi_data/shit_data.txt"

class ShiScraper(AuctionScraper):
    """
    Parses shi_data.txt and emits one (artist, price) pair per individual sale.

    The file is tab-separated with 23 columns. Artist names live in column 1
    and prices in column 7. Names contain spaces, so naive whitespace
    splitting breaks — splitting on tabs avoids that. Malformed rows (those
    without the expected column count or with an unparseable price) are
    skipped.
    """

    ARTIST_COL = 1
    PRICE_COL = 7
    EXPECTED_COLS = 23

    def __init__(self, path: str = PATH):
        self.path = path

    def get_data(
            self,
            acceptance_function: Optional[Callable[[str], bool]] = None
        ) -> tuple[list, np.ndarray]:
        names: list[str] = []
        prices: list[float] = []

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)  # skip header
            for line in f:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != self.EXPECTED_COLS:
                    continue
                artist = fields[self.ARTIST_COL].strip()
                price_str = fields[self.PRICE_COL].strip()
                if not artist or not price_str:
                    continue
                if acceptance_function is not None and not acceptance_function(artist):
                    continue
                try:
                    price = float(price_str)
                except ValueError:
                    continue
                if price <= 0:
                    continue
                names.append(artist)
                prices.append(price)

        return names, np.array(prices)

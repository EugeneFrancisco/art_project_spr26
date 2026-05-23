"""
Scraper for the shi_data.txt auction history file.
"""

from collections import defaultdict

import numpy as np

from src.data_utils.data_wrapper import AuctionScraper


class ShiScraper(AuctionScraper):
    """
    Parses shi_data.txt and reports each artist's median sale price.

    The file is tab-separated with 23 columns. Artist names live in column 1
    and prices in column 7. Names contain spaces, so naive whitespace
    splitting breaks — splitting on tabs avoids that. Malformed rows (those
    without the expected column count or with an unparseable price) are
    skipped.
    """

    ARTIST_COL = 1
    PRICE_COL = 7
    EXPECTED_COLS = 23

    def __init__(self, path: str):
        self.path = path

    def get_data(self) -> tuple[list, np.ndarray]:
        prices_by_artist = defaultdict(list)

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
                try:
                    price = float(price_str)
                except ValueError:
                    continue
                prices_by_artist[artist].append(price)

        names = sorted(prices_by_artist)
        medians = np.array([np.median(prices_by_artist[n]) for n in names])
        return names, medians

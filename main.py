import time
import os
import requests
import xml.etree.ElementTree as ET
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Requires: pip install vaderSentiment

# Assuming the two files are in the same directory or importable
from roostoo_api import roostooAPI
from binance_api import cryptoAPI  # We'll use this for potential real-market reference if needed, but mainly roostoo for trading


class SentimentTradingStrategy:
    def __init__(self, interval_seconds=10, sell_threshold=-0.5, buy_threshold=0.5, sell_proportion=0.5,
                 buy_proportion=0.1):
        self.api = roostooAPI()
        self.binance_api = cryptoAPI()  # Initialized with Binance, can be used for real data if needed

        self.cryptos = self.api.get_all_ticker_id()
        if self.cryptos is None or len(self.cryptos) != 66:
            raise ValueError("Could not retrieve the 66 cryptos from Roostoo API or incorrect count.")

        self.bases = [pair.split('/')[0] for pair in self.cryptos]

        # Updated mapping of crypto symbols to their common full names (lowercased) for sentiment analysis
        self.crypto_names = {
            '1000CHEEMS': 'cheems',
            'AAVE': 'aave',
            'ADA': 'cardano',
            'APT': 'aptos',
            'ARB': 'arbitrum',
            'ASTER': 'aster',
            'AVAX': 'avalanche',
            'AVNT': 'avantis',
            'BIO': 'bio protocol',
            'BMT': 'bubblemaps',
            'BNB': 'bnb',
            'BONK': 'bonk',
            'BTC': 'bitcoin',
            'CAKE': 'pancake swap',
            'CFX': 'conflux',
            'CRV': 'curve dao',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'EDEN': 'eden network',
            'EIGEN': 'eigenlayer',
            'ENA': 'ethena',
            'ETH': 'ethereum',
            'FET': 'artificial superintelligence alliance',
            'FIL': 'filecoin',
            'FLOKI': 'floki',
            'FORM': 'form',
            'HBAR': 'hedera',
            'HEMI': 'hemi',
            'ICP': 'internet computer',
            'LINEA': 'linea',
            'LINK': 'chainlink',
            'LISTA': 'lista dao',
            'LTC': 'litecoin',
            'MIRA': 'mira',
            'NEAR': 'near protocol',
            'ONDO': 'ondo finance',
            'OPEN': 'openledger',
            'PAXG': 'pax gold',
            'PENDLE': 'pendle',
            'PENGU': 'pudgy penguins',
            'PEPE': 'pepe',
            'PLUME': 'plume',
            'POL': 'polygon',
            'PUMP': 'pump fun',
            'S': 's',
            'SEI': 'sei',
            'SHIB': 'shiba inu',
            'SOL': 'solana',
            'SOMI': 'somnia',
            'STO': 'stakestone',
            'SUI': 'sui',
            'TAO': 'bittensor',
            'TON': 'toncoin',
            'TRUMP': 'trump',
            'TRX': 'tron',
            'TUT': 'tutorial',
            'UNI': 'uniswap',
            'VIRTUAL': 'virtuals protocol',
            'WIF': 'dogwifhat',
            'WLD': 'worldcoin',
            'WLFI': 'world liberty financial',
            'XLM': 'stellar',
            'XPL': 'plasma',
            'XRP': 'xrp',
            'ZEC': 'zcash',
            'ZEN': 'horizen'
        }

        self.seen_titles = set()
        self.interval_seconds = interval_seconds
        self.sell_threshold = sell_threshold  # To be tuned
        self.buy_threshold = buy_threshold  # To be tuned
        self.sell_proportion = sell_proportion  # To be tuned
        self.buy_proportion = buy_proportion  # To be tuned

        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def get_latest_news_titles():
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
        try:
            response = requests.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            titles = [item.find("title").text for item in root.findall("./channel/item") if
                      item.find("title") is not None]
            return titles
        except Exception as e:
            print(f"Error fetching news from CoinDesk: {e}")
            return []

    def get_sentiment_scores(self, title, bases):
        """
        Use VADER for advanced sentiment analysis.
        VADER returns a compound score from -1 (negative) to 1 (positive).
        """
        scores = {b: 0.0 for b in bases}
        title_lower = title.lower()

        sentiment = self.analyzer.polarity_scores(title)
        overall_score = sentiment['compound']

        # Assign score only to mentioned cryptos
        for b in bases:
            name = self.crypto_names.get(b, b.lower())
            if name in title_lower or b.lower() in title_lower:
                scores[b] = overall_score

        return scores

    def run(self):
        print("Starting sentiment-based trading strategy on Roostoo with VADER sentiment analysis...")
        while True:
            titles = self.get_latest_news_titles()
            new_titles = [t for t in titles if t not in self.seen_titles]

            for title in new_titles:
                print(f"New news detected: {title}")
                scores = self.get_sentiment_scores(title, self.bases)

                balance_resp = self.api.get_balance()
                if balance_resp is None:
                    print("Failed to get balance, skipping.")
                    continue

                # Assuming balance response format: {'Data': [{'asset': 'BTC', 'free': '1.0'}, ...]}
                balances = {b['asset']: float(b['free']) for b in balance_resp.get('Data', [])}
                usd_balance = balances.get('USD', 0.0)

                for base, score in scores.items():
                    if score == 0.0:
                        continue  # No impact

                    pair = f"{base}/USD"
                    current_amount = balances.get(base, 0.0)

                    ticker = self.api.get_ticker(pair)
                    if ticker is None:
                        print(f"Failed to get ticker for {pair}, skipping.")
                        continue

                    # Assuming ticker format: {'Data': {'BTC/USD': {'lastPrice': '60000'}}}
                    price_data = ticker.get('Data', {}).get(pair, {})
                    price = float(price_data.get('lastPrice', 0.0))
                    if price == 0.0:
                        continue

                    if score < self.sell_threshold and current_amount > 0:
                        sell_qty = current_amount * self.sell_proportion
                        print(f"Selling {sell_qty} of {base} due to low sentiment ({score}).")
                        self.api.place_order(base, 'SELL', sell_qty, order_type='MARKET')

                    elif score > self.buy_threshold and usd_balance > 0:
                        spend_amount = usd_balance * self.buy_proportion
                        buy_qty = spend_amount / price
                        print(f"Buying {buy_qty} of {base} due to high sentiment ({score}).")
                        self.api.place_order(base, 'BUY', buy_qty, order_type='MARKET')

            self.seen_titles.update(new_titles)
            time.sleep(self.interval_seconds)


if __name__ == "__main__":
    # Ensure ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY are set in environment variables
    if not os.environ.get("ROOSTOO_API_KEY") or not os.environ.get("ROOSTOO_SECRET_KEY"):
        raise ValueError("Please set ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY environment variables.")

    strategy = SentimentTradingStrategy(interval_seconds=10)  # Adjust parameters as needed
    strategy.run()

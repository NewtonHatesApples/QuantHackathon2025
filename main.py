import time
import os
import requests
import xml.etree.ElementTree as ET
import json
import random

from bs4 import BeautifulSoup
from roostoo_api import roostooAPI
from binance_api import cryptoAPI  # We'll use this for potential real-market reference if needed, but mainly roostoo for trading


class SentimentTradingStrategy:
    def __init__(self, interval_seconds=10, sell_threshold=-0.6, buy_threshold=0.6, sell_proportion=0.8,
                 buy_proportion=0.01):
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

        self.seen_links = set()
        self.interval_seconds = interval_seconds
        self.sell_threshold = sell_threshold  # To be tuned
        self.buy_threshold = buy_threshold  # To be tuned
        self.sell_proportion = sell_proportion  # To be tuned
        self.buy_proportion = buy_proportion  # To be tuned

        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/118.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        ]

        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/search?q=coindesk',
            'Connection': 'keep-alive',
        }

    def get_latest_news(self):
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
        headers = self.headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = []
            for item in root.findall("./channel/item"):
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                description = item.find("description").text if item.find("description") is not None else ""
                pubdate = item.find("pubDate").text if item.find("pubDate") is not None else ""
                items.append({"title": title, "link": link, "description": description, "pubdate": pubdate})
            return items
        except Exception as e:
            print(f"Error fetching news from CoinDesk: {e}")
            return []

    def get_article_content(self, url):
        headers = self.headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(p.text.strip() for p in paragraphs if p.text.strip())
            return content
        except Exception as e:
            print(f"Error fetching article content from {url}: {e}")
            return ""

    def get_sentiment_scores(self, title, content, bases):
        """
        Use xAI Grok API for sentiment analysis.
        """
        system_prompt = "You are a crypto sentiment analyst. Respond only with a JSON object where keys are cryptocurrency symbols and values are sentiment scores from -1 to 1."
        user_prompt = f"Analyze this crypto news for sentiment impact on these cryptocurrencies: {', '.join(bases)} (with names: {', '.join([self.crypto_names.get(b, b) for b in bases])}).\n\nTitle: {title}\n\nContent: {content}\n\nFor each symbol, score from -1 (very negative) to 1 (very positive) based on potential price impact. 0 if unrelated."

        headers = {
            "Authorization": f"Bearer {os.environ.get('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "grok-4",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 4096
        }

        try:
            response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            scores_str = result['choices'][0]['message']['content']
            scores = json.loads(scores_str)
            # Ensure all bases are present
            for b in bases:
                if b not in scores:
                    scores[b] = 0.0
            return scores
        except Exception as e:
            print(f"Error getting sentiment scores from xAI API: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text if e.response else 'N/A'}")
            return {b: 0.0 for b in bases}

    def run(self):
        print("Starting sentiment-based trading strategy on Roostoo with LLM sentiment analysis...")
        while True:
            items = self.get_latest_news()
            new_items = [i for i in items if i['link'] not in self.seen_links]

            for item in new_items:
                title = item['title']
                print(f"\nNew news detected: {title}")
                content = self.get_article_content(item['link'])
                scores = self.get_sentiment_scores(title, content, self.bases)

                balance_resp = self.api.get_balance()
                if balance_resp is None:
                    print("Failed to get balance, skipping.")
                    continue

                # Assuming balance response format: {'Data': [{'asset': 'BTC', 'free': '1.0'}, ...]}
                balances = dict()
                wallet = balance_resp['SpotWallet']
                for crypto, bal in wallet.items():
                    balances[crypto] = bal["Free"]
                usd_balance = balances.get('USD', 0.0)

                for base, score in scores.items():
                    if score == 0.0:
                        continue  # No impact

                    pair = f"{base}/USD"
                    current_amount = balances.get(base, 0.0)
                    price_precision, amount_precision, mini_order = self.api.get_pair_info(pair)

                    ticker = self.api.get_ticker(pair)
                    if ticker is None:
                        print(f"Failed to get ticker for {pair}, skipping.")
                        continue

                    # Assuming ticker format: {'Data': {'BTC/USD': {'lastPrice': '60000'}}}
                    price_data = ticker.get('Data', {}).get(pair, {})
                    price = float(price_data.get('LastPrice', 0.0))
                    if price == 0.0:
                        continue

                    if score <= self.sell_threshold and current_amount > 0:
                        sell_qty = round(current_amount * self.sell_proportion, amount_precision)
                        if sell_qty * price >= mini_order:
                            print(f"Selling {sell_qty} of {base} due to low sentiment ({score}).")
                            self.api.place_order(base, 'SELL', sell_qty, order_type='MARKET')
                        else:
                            print(f"NOT buying {sell_qty} of {base} because order total amount too low.")

                    elif score >= self.buy_threshold and usd_balance > 0:
                        spend_amount = usd_balance * self.buy_proportion
                        buy_qty = round(spend_amount / price, amount_precision)
                        if buy_qty * price >= mini_order:
                            print(f"Buying {buy_qty} of {base} due to high sentiment ({score}).")
                            self.api.place_order(base, 'BUY', buy_qty, order_type='MARKET')
                        else:
                            print(f"NOT buying {buy_qty} of {base} because order total amount too low.")

            self.seen_links.update([i['link'] for i in new_items])
            time.sleep(self.interval_seconds)


if __name__ == "__main__":
    strategy = SentimentTradingStrategy()  # Adjust parameters as needed
    strategy.run()

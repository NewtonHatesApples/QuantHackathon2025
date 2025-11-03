import ccxt
import datetime
from dateutil.relativedelta import relativedelta


class CryptoAPI:
    def __init__(self, exchange_name='binance'):
        """
        Initialize the API with a specific exchange.
        :param exchange_name: Name of the exchange (e.g., 'binance', 'coinbase', etc.)
        """
        self.exchange = getattr(ccxt, exchange_name)()
        if not self.exchange.has['fetchTicker'] or not self.exchange.has['fetchOHLCV']:
            raise ValueError(f"Exchange {exchange_name} does not support required features.")

    def get_current_info(self, symbol):
        """
        Fetch current important info for the given symbol.
        :param symbol: Symbol in the format 'BTC/USD' or 'BTC/USDT', etc.
        :return: Dictionary with current price, volume (24h quote volume), ask, bid, etc.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'last_price': ticker['last'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'volume_24h_base': ticker['baseVolume'],  # Volume in base currency (e.g., BTC)
            'volume_24h_quote': ticker['quoteVolume'],  # Volume in quote currency (e.g., USD)
            'ask': ticker['ask'],
            'bid': ticker['bid'],
            'ask_volume': ticker['askVolume'],
            'bid_volume': ticker['bidVolume'],
            'timestamp': ticker['timestamp'],
        }

    def get_historical_data(self, symbol, start_date=None, end_date=None, interval='1d'):
        """
        Fetch historical price-volume data (OHLCV) for the given symbol and timeframe.
        Note: Historical ask-bid data is not typically available through standard exchange APIs,
        as order book snapshots are real-time only. This method returns OHLCV data (open, high, low,
        close, volume) which includes price and volume history. High/low can serve as proxies for
        ask/bid ranges within each candle.

        Date handling:
        - If both start_date and end_date are None, use the most recent 5-year period.
        - If start_date is None but end_date is given, set start_date to 5 years before end_date.
        - If end_date is None but start_date is given, set end_date to 5 years after start_date.
        - Dates should be in 'YYYY-MM-DD' format.

        Interval values: Common options include '1m' (1 minute), '3m', '5m', '15m', '30m', '1h' (1 hour),
        '2h', '4h', '6h', '8h', '12h', '1d' (1 day), '3d', '1w' (1 week), '1M' (1 month). Availability
        depends on the exchange; check exchange.timeframes for supported values.

        :param symbol: Symbol in the format 'BTC/USD' or 'BTC/USDT', etc.
        :param start_date: Start date as 'YYYY-MM-DD' (optional).
        :param end_date: End date as 'YYYY-MM-DD' (optional).
        :param interval: Timeframe interval (e.g., '1d') (optional, default '1d').
        :return: List of lists, each [timestamp (ms), open, high, low, close, volume].
        """
        today = datetime.date.today()

        if start_date is None and end_date is None:
            end_date = today
            start_date = end_date - relativedelta(years=5)
        elif start_date is None:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            start_date = end_date - relativedelta(years=5)
        elif end_date is None:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = start_date + relativedelta(years=5)
        else:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

        # Convert start_date to milliseconds timestamp
        since = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp() * 1000)

        # Fetch OHLCV data (may need to paginate if timeframe is large)
        ohlcv = []
        while True:
            data = self.exchange.fetch_ohlcv(symbol, interval, since, limit=1000)
            if not data:
                break
            ohlcv.extend(data)
            since = data[-1][0] + 1  # Next batch starts after last timestamp
            last_date = datetime.datetime.fromtimestamp(data[-1][0] / 1000).date()
            if last_date >= end_date:
                break

        # Filter to end_date if needed
        end_timestamp = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp() * 1000)
        ohlcv = [candle for candle in ohlcv if candle[0] <= end_timestamp]

        return ohlcv

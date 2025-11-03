import requests
import os
import time
import hmac
import hashlib

BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.environ.get("ROOSTOO_API_KEY")
SECRET_KEY = os.environ.get("ROOSTOO_SECRET_KEY")

SERVER_TIME_URL = BASE_URL + "/v3/serverTime"
EXCHANGE_URL = BASE_URL + "/v3/exchangeInfo"
TICKER_URL = BASE_URL + "/v3/ticker"
BALANCE_URL = BASE_URL + "/v3/balance"
PENDING_URL = BASE_URL + "/v3/pending_count"
PLACE_ORDER_URL = BASE_URL + "/v3/place_order"
QUERY_ORDER_URL = BASE_URL + "/v3/query_order"
CANCEL_ORDER_URL = BASE_URL + "/v3/cancel_order"
# --------------------------------------------------------

def _get_timestamp():
    """Returns a 13-digit millisecond timestamp as a string."""
    return str(int(time.time() * 1000))


def _get_signed_headers(payload: dict | None = None):
    """
    Creates a signature for a given payload (dict) and returns
    the correct headers for a SIGNED (RCL_TopLevelCheck) request.
    """
    # 1. Add timestamp to the payload
    if payload is None:
        payload = {}
    payload['timestamp'] = _get_timestamp()

    # 2. Sort keys and create the totalParams string
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{key}={payload[key]}" for key in sorted_keys)

    # 3. Create HMAC-SHA256 signature
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # 4. Create headers
    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }

    return headers, payload, total_params

# --- Now we can define functions for each API call ---

def check_server_time():
    """Checks server time. (Auth: RCL_NoVerification)"""
    try:
        response = requests.get(SERVER_TIME_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking server time: {e}")
        return None


def get_exchange_info():
    """Gets exchange info. (Auth: RCL_NoVerification)"""
    try:
        response = requests.get(EXCHANGE_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting exchange info: {e}")
        return None


def get_ticker(pair=None):
    """Gets market ticker. (Auth: RCL_TSCheck)"""
    params = {
        'timestamp': _get_timestamp()
    }
    if pair:
        params['pair'] = pair

    try:
        response = requests.get(TICKER_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting ticker: {e}")
        return None


def get_balance():
    """Gets account balance. (Auth: RCL_TopLevelCheck)"""
    # 1. Get signed headers and the payload (which now includes timestamp)
    # For a GET request with no params, the payload is just the timestamp
    headers, payload, total_params_string = _get_signed_headers()

    try:
        # 2. Send the request
        # In a GET request, the payload is sent as 'params'
        response = requests.get(BALANCE_URL, headers=headers, params=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting balance: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def get_pending_count():
    """Gets pending order count. (Auth: RCL_TopLevelCheck)"""
    headers, payload, total_params_string = _get_signed_headers(payload={})

    try:
        response = requests.get(PENDING_URL, headers=headers, params=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pending count: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    """
    Places a new order with improved flexibility and safety checks.

    Args:
        pair_or_coin (str): The asset to trade (e.g., "BTC" or "BTC/USD").
        side (str): "BUY" or "SELL".
        quantity (float or int): The amount to trade.
        price (float, optional): The price for a LIMIT order. Defaults to None.
        order_type (str, optional): "LIMIT" or "MARKET". Auto-detected if not provided.
    """
    print(f"\n--- Placing a new order for {quantity} {pair_or_coin} ---")

    # 1. Determine the full pair name
    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    # 2. Auto-detect order_type if it's not specified
    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"
        print(f"Auto-detected order type: {order_type}")

    # 3. Validate parameters to prevent errors
    if order_type == 'LIMIT' and price is None:
        print("Error: LIMIT orders require a 'price' parameter.")
        return None
    if order_type == 'MARKET' and price is not None:
        print("Warning: Price is provided for a MARKET order and will be ignored by the API.")

    # 4. Create the request payload
    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)

    # 5. Get signed headers and the final request body
    headers, payload, total_params = _get_signed_headers(payload)

    # 6. Send the request
    try:
        response = requests.post(PLACE_ORDER_URL, headers=headers, data=total_params)
        response.raise_for_status()
        print(f"API Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def query_order(order_id=None, pair=None, pending_only=None):
    """Queries orders. (Auth: RCL_TopLevelCheck)"""
    payload = {}
    if order_id and pair:
        print("order_id and pair cannot be sent together.")
        return None
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:  # Docs say order_id and pair cannot be sent together
        payload['pair'] = pair
        if pending_only is not None:
            # Docs specify STRING_BOOL
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

    headers, final_payload, total_params_string = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        response = requests.post(QUERY_ORDER_URL, headers=headers, data=total_params_string)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def cancel_order(order_id=None, pair=None):
    """Cancels orders. (Auth: RCL_TopLevelCheck)"""
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:  # Docs say only one is allowed
        payload['pair'] = pair
    # If neither is sent, it cancels all

    headers, final_payload, total_params_string = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        response = requests.post(CANCEL_ORDER_URL, headers=headers, data=total_params_string)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error canceling order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def get_all_ticker_id():
    """Get all ticker ids."""
    resp = get_ticker()
    data = resp.get('Data')
    if data is not None:
        return list(data.keys())
    else:
        print(f"Failed to get ticker id from API! Error: {resp['ErrMsg']}")
        return None


if __name__ == "__main__":
    print(get_all_ticker_id())
    # Example 1: Place a LIMIT order (by providing a price)
    # The function will correctly identify this as a LIMIT order.
    place_order(pair_or_coin="BTC", side="SELL", quantity=0.01, price=99000)

    # Example 2: Place a MARKET order (by not providing a price)
    # The function will correctly identify this as a MARKET order.
    place_order(pair_or_coin="BNB/USD", side="BUY", quantity=10)

    # Example 3: Invalid order (LIMIT without a price)
    # The function will catch this error before sending the request.
    place_order(pair_or_coin="ETH", side="BUY", quantity=0.5, order_type="LIMIT") # Explicitly set, but no price given

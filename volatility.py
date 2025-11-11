import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from binance_api import cryptoAPI
from roostoo_api import roostooAPI

# Coin list from HORUS API Documentation -> From Roostoo platform
coin_list = roostooAPI().get_all_ticker_id()

# Initializing the code provided by our goat
api = cryptoAPI('binance')


def convert_to_df(data: list[list]) -> pd.DataFrame:
    """
    Converts a list of lists containing financial data into a pandas DataFrame.

    Each inner list should represent a row with values in the order:
    [timestamp (ms), open, high, low, close, volume]

    The timestamp is converted from milliseconds to a datetime object.

    Args:
        data (list[list]): The input data as a list of lists.

    Returns:
        pd.DataFrame: The resulting DataFrame with appropriate columns.
    """
    columns = ['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(data, columns=columns)

    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df.drop(columns=['timestamp_ms'])  # Remove the original ms column
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Reorder columns

    return df


def calculate_rolling_volatility(df, window_days=7):
    """
    Calculate rolling volatility for a given window
    """
    if df.empty:
        return df

    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    df['daily_return'].dropna(inplace=True)

    # Calculate rolling volatility (annualized)
    df[f'vol_{window_days}d'] = df['daily_return'].rolling(window=window_days).std() * np.sqrt(365)

    return df


def comprehensive_volatility_analysis(df):
    """
    Perform comprehensive volatility analysis including 1-week and 2-week rolling volatility
    """
    if df.empty:
        return df, {}

    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()

    # Calculate rolling volatilities
    df = calculate_rolling_volatility(df, window_days=7)  # 1-week volatility
    df = calculate_rolling_volatility(df, window_days=14)  # 2-week volatility

    # Calculate additional volatility metrics (maybe can use something other than vol_7d. Honestly, idk what relevance these others have I just AId this)
    volatility_metrics = {
        'current_price': df['close'].iloc[-1],
        'annual_volatility': df['daily_return'].std() * np.sqrt(365),
        'vol_7d_current': df['vol_7d'].iloc[-1] if not pd.isna(df['vol_7d'].iloc[-1]) else np.nan,
        'vol_7d_avg': df['vol_7d'].mean(),
        'vol_7d_max': df['vol_7d'].max(),
        'vol_14d_current': df['vol_14d'].iloc[-1] if not pd.isna(df['vol_14d'].iloc[-1]) else np.nan,
        'vol_14d_avg': df['vol_14d'].mean(),
        'vol_14d_max': df['vol_14d'].max(),
        'volatility_ratio_7d_14d': (df['vol_7d'].iloc[-1] / df['vol_14d'].iloc[-1])
        if not pd.isna(df['vol_7d'].iloc[-1]) and not pd.isna(df['vol_14d'].iloc[-1]) and df['vol_14d'].iloc[-1] != 0
        else np.nan,
        'data_points': len(df),
        'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    }

    return df, volatility_metrics


def df_getter():
    ucoin_list = []
    for i in coin_list:
        i = i.replace("USD", "USDT")
        ucoin_list.append(i)

    # Dictionary that has the key as the ticker of the coin and the value to be the dataframe of its historical data
    data_dict = {}
    for coin in ucoin_list:
        data = convert_to_df(api.get_historical_data(coin, start_date='2024-11-10', end_date='2025-11-10', interval='15m'))
        data_dict[coin] = data

    # Putting into new dictionary
    metrics_dict = {}
    new_data_dict = {}

    # new_data_dict : Key: Ticker, Value: DataFrame with the ticker's OHLCV, Daily Return, 7 day volatility and 14 day volatility
    # metrics_df : Rows: Analytics of rolling volatility, Columns: Ticker
    for key, value in data_dict.items():
        print(key, '\n', value)
        new_data, metrics = comprehensive_volatility_analysis(value)
        metrics_dict[key] = metrics
        new_data_dict[key] = new_data
    metrics_df = pd.DataFrame(metrics_dict)
    return metrics_df, new_data_dict


def plot_7d_volatility(metrics_df, top_n=20, figsize=(15, 10)):
    """
    Plot the 7-day volatility for each ticker in metrics_df

    Parameters:
    - metrics_df: DataFrame containing volatility metrics
    - top_n: Number of top volatile coins to display (default: 20)
    - figsize: Figure size (default: (15, 10))
    """
    # Apparently this looks pretty cool so tried it and was right
    plt.style.use('seaborn-v0_8')

    vol_data = metrics_df.loc['vol_7d_avg'].astype(float)
    vol_data = vol_data.dropna()

    if vol_data.empty:
        print("No volatility data available to plot.")
        return

    # Sorting by volatility
    top_vol = vol_data.nlargest(top_n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Bar chart of top N most volatile coins
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_vol)))
    bars = ax1.bar(range(len(top_vol)), top_vol.values * 100, color=colors, alpha=0.7)
    ax1.set_title(f'Top {top_n} Most Volatile Cryptocurrencies (7-Day Rolling Volatility)', fontsize=16,
                  fontweight='bold')
    ax1.set_ylabel('7-Day Volatility (%)', fontsize=12)
    ax1.set_xlabel('Cryptocurrency', fontsize=12)

    # Set x-axis labels with rotated text for readability
    tickers_clean = [ticker.replace('/USDT', '') for ticker in top_vol.index]
    ax1.set_xticks(range(len(top_vol)))
    ax1.set_xticklabels(tickers_clean, rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Distribution of all 7-day volatilities (Very messy I know)
    ax2.hist(vol_data.values * 100, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(vol_data.mean() * 100, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {vol_data.mean() * 100:.1f}%')
    ax2.axvline(vol_data.median() * 100, color='orange', linestyle='--', linewidth=2,
                label=f'Median: {vol_data.median() * 100:.1f}%')

    ax2.set_title('Distribution of 7-Day Volatility Across All Cryptocurrencies', fontsize=16, fontweight='bold')
    ax2.set_xlabel('7-Day Volatility (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_7d_volatility_linechart(new_data_dict, figsize=(15, 10), top_n=15):
    """
    Plot line charts for the 7-day volatility of each ticker

    Parameters:
    - new_data_dict: Dictionary with tickers as keys and DataFrames with volatility data as values
    - figsize: Figure size (default: (15, 10))
    - top_n: Number of top volatile coins to highlight (default: 15)
    """
    # Cool style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate average current 7-day volatility for each ticker to determine top N
    current_volatilities = {}
    valid_tickers = {}

    for ticker, df in new_data_dict.items():
        if not df.empty and 'vol_7d' in df.columns:
            # Remove NaN values from volatility data
            vol_data = df['vol_7d'].dropna()
            if not vol_data.empty:
                valid_tickers[ticker] = vol_data
                current_volatilities[ticker] = vol_data.iloc[-1] if not pd.isna(vol_data.iloc[-1]) else 0

    if not valid_tickers:
        print("No valid volatility data found.")
        return

    # Sort by current volatility to get top N
    sorted_tickers = sorted(current_volatilities.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [ticker for ticker, vol in sorted_tickers[:top_n]]
    other_tickers = [ticker for ticker, vol in sorted_tickers[top_n:]]

    # This I have no idea what it does but I guess it makes the color pop
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_tickers)))

    # Plot top n tickers with the cool colours
    for i, ticker in enumerate(top_tickers):
        vol_data = valid_tickers[ticker]
        clean_ticker = ticker.replace('/USDT', '')
        ax.plot(vol_data.index, vol_data.values * 100,
                label=clean_ticker, color=colors[i], linewidth=2, alpha=0.8)

    # Plot other tickers but in light gray to distinguish
    for ticker in other_tickers:
        vol_data = valid_tickers[ticker]
        ax.plot(vol_data.index, vol_data.values * 100,
                color='lightgray', linewidth=0.5, alpha=0.3)

    # Customize the plot
    ax.set_title(f'7-Day Rolling Volatility Over Time\n(Top {top_n} highlighted)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('7-Day Volatility (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add legend (place it outside the plot if there are many lines)
    if len(top_tickers) <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=min(5, len(top_tickers) // 2 + 1), frameon=True)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(valid_tickers)} cryptocurrencies")
    print(f"Highlighted top {len(top_tickers)} most volatile coins")

    return fig


# Alternative version with subplots for better clarity
def plot_7d_volatility_subplots(new_data_dict, figsize=(20, 12), rows=3, cols=5):
    """
    Plot 7-day volatility in a grid of subplots for better individual visualization
    """
    # Get all valid tickers with volatility data
    valid_tickers = {}
    for ticker, df in new_data_dict.items():
        if not df.empty and 'vol_7d' in df.columns:
            vol_data = df['vol_7d'].dropna()
            if not vol_data.empty:
                valid_tickers[ticker] = vol_data

    if not valid_tickers:
        print("No valid volatility data found.")
        return

    # Sort by current volatility (highest first)
    sorted_tickers = sorted(valid_tickers.items(),
                            key=lambda x: x[1].iloc[-1] if not x[1].empty else 0,
                            reverse=True)

    # Calculate grid size
    total_plots = min(rows * cols, len(sorted_tickers))

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    # Plot each ticker in its own subplot
    for i, (ticker, vol_data) in enumerate(sorted_tickers[:total_plots]):
        ax = axes[i]
        clean_ticker = ticker.replace('/USDT', '')
        current_vol = vol_data.iloc[-1] * 100 if not vol_data.empty else 0

        ax.plot(vol_data.index, vol_data.values * 100, color='blue', linewidth=1.5)
        ax.set_title(f'{clean_ticker}\n(Current: {current_vol:.1f}%)', fontsize=10)
        ax.set_ylabel('Volatility (%)', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(total_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('7-Day Rolling Volatility - Individual Cryptocurrencies',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    return fig


# Simple version for just a few selected tickers
def plot_selected_tickers_volatility(new_data_dict, selected_tickers, figsize=(12, 8)):
    """
    Plot 7-day volatility for specific selected tickers
    """
    plt.figure(figsize=figsize)

    for ticker in selected_tickers:
        if ticker in new_data_dict and not new_data_dict[ticker].empty:
            df = new_data_dict[ticker]
            if 'vol_7d' in df.columns:
                vol_data = df['vol_7d'].dropna()
                clean_ticker = ticker.replace('/USDT', '')
                plt.plot(vol_data.index, vol_data.values * 100,
                         label=clean_ticker, linewidth=2, alpha=0.8)

    plt.title('7-Day Rolling Volatility - Selected Cryptocurrencies',
              fontsize=14, fontweight='bold')
    plt.ylabel('7-Day Volatility (%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Get the data first
    metrics_df, new_data_dict = df_getter()

    # Plot all tickers in one line chart
    plot_7d_volatility_linechart(new_data_dict, top_n=15)

    # Plot in subplot grid
    plot_7d_volatility_subplots(new_data_dict, rows=4, cols=5)

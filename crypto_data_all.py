"""
Crypto Hourly Data Downloader with Strict Contiguity Validation
----------------------------------------------------------------
This script downloads hourly OHLCV data for multiple cryptocurrency pairs from
Bitstamp (or any CCXT-supported exchange). It performs rigorous checks to ensure
no single hour is missing within the downloaded range and that the data
realistically covers the requested date span. After validation, the full
contiguous downloaded range is exported; downstream scripts then compute
log-returns and `liquidity.py` applies the final liquid analysis window.
"""

import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from typing import Optional  # For Python < 3.10 compatibility
from config_loader import load_config, get_data_dir, get_download_range

# ------------------------------------------------------------------------------
# CONFIGURATION SECTION
# ------------------------------------------------------------------------------
CONFIG = load_config()
# Output directory where CSV files will be saved.
# Controlled by config.yaml -> paths.data_dir
OUTPUT_DIR = get_data_dir(CONFIG)

# Exchange identifier string as recognized by the ccxt library.
EXCHANGE_ID = 'bitstamp'

# Timeframe for the OHLCV candles. '1h' requests hourly bars.
TIMEFRAME = '1h'

# Dictionary mapping trading pair symbols to the earliest date Bitstamp
# officially started trading that pair. These dates are based on public
# announcements and serve as the starting point for data download.
PAIR_START_DATES = {
    'BTC/USD': '2011-08-18',
    'XRP/USD': '2017-01-17',
    'LTC/USD': '2017-06-19',
    'ETH/USD': '2017-08-17',
    'LINK/USD': '2020-10-19',
    'ADA/USD': '2021-11-24',
    'DOGE/USD': '2022-12-22',
}

# Date range override from config.yaml:
# - download.from: global start date override (optional)
# - download.to: global end date override (optional)
GLOBAL_FROM_STR, END_DATE_STR = get_download_range(CONFIG)

# Create the output directory if it does not already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# FUNCTION: download_hourly_data
# ------------------------------------------------------------------------------
def download_hourly_data(symbol: str,
                         start_str: str,
                         end_str: str,
                         exchange_id: str,
                         timeframe: str) -> Optional[pd.DataFrame]:
    """
    Download and validate contiguous hourly OHLCV data for a single trading pair.

    The function fetches data page by page from the exchange API, assembles it
    into a pandas DataFrame, and performs a series of integrity checks:

        1. No duplicate timestamps.
        2. No missing hours between the earliest and latest candle retrieved.
        3. Realistic boundary conditions:
           - Data must not start later than 23 hours after the requested start date
             (allowing for pairs that began trading later on their launch day).
           - If the end date is today, the latest candle must be within ~2 hours
             of the current UTC hour (accommodating minor API delays).
           - If the end date is in the past, the latest candle must reach 23:00
             of that day.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USD').
    start_str : str
        Start date in 'YYYY-MM-DD' format (UTC).
    end_str : str
        End date in 'YYYY-MM-DD' format (UTC).
    exchange_id : str
        CCXT exchange identifier (e.g., 'bitstamp').
    timeframe : str
        OHLCV timeframe (e.g., '1h').

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame with columns: timestamp, datetime, open, high, low, close,
        volume. Returns None if any validation fails.
    """
    # --------------------------------------------------------------------------
    # Step 1: Instantiate the exchange object.
    # --------------------------------------------------------------------------
    # Dynamically get the exchange class from ccxt using the provided ID string.
    exchange_class = getattr(ccxt, exchange_id)
    # Create an instance with rate limiting enabled (prevents API bans).
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}   # Ensure we query the spot market.
    })

    # --------------------------------------------------------------------------
    # Step 2: Convert start/end strings to millisecond UTC timestamps.
    # --------------------------------------------------------------------------
    # parse8601 converts an ISO8601-like string to a Unix timestamp in ms.
    # We append 'T00:00:00Z' to start and 'T23:59:59Z' to end to define full days.
    since_ms = exchange.parse8601(f'{start_str}T00:00:00Z')
    until_ms = exchange.parse8601(f'{end_str}T23:59:59Z')

    # This list will collect all raw OHLCV arrays returned by the API.
    all_ohlcv = []
    # current_ms tracks the starting timestamp for the next page of data.
    current_ms = since_ms

    print(f"Downloading {symbol} ({timeframe}) from {start_str} to {end_str}...")

    # --------------------------------------------------------------------------
    # Step 3: Paginated download loop.
    # --------------------------------------------------------------------------
    # We fetch up to 1000 candles per request (maximum allowed by most exchanges).
    while current_ms < until_ms:
        try:
            # Request OHLCV data from the exchange.
            ohlcv_chunk = exchange.fetch_ohlcv(
                symbol, timeframe, since=current_ms, limit=1000
            )
            # If the returned list is empty, we have reached the end of available data.
            if not ohlcv_chunk:
                break

            # Append the new chunk to our master list.
            all_ohlcv.extend(ohlcv_chunk)

            # Advance the pagination pointer to the timestamp immediately after
            # the last candle we just received. This ensures we get the next
            # contiguous segment.
            current_ms = ohlcv_chunk[-1][0] + 1

            print(f"  Fetched {len(ohlcv_chunk)} rows, total {len(all_ohlcv)}")
            # Pause briefly to respect exchange rate limits (200 ms is safe).
            time.sleep(0.2)

        except Exception as e:
            # If any error occurs (network, rate limit, etc.), wait longer and retry.
            print(f"  Error: {e}, retrying in 5 seconds...")
            time.sleep(5)

    # If no data was retrieved at all, return None.
    if not all_ohlcv:
        print("  No data retrieved.")
        return None

    # --------------------------------------------------------------------------
    # Step 4: Build a pandas DataFrame from the raw data.
    # --------------------------------------------------------------------------
    # Each element in all_ohlcv is a list: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    # Convert the millisecond timestamp column to UTC datetime objects.
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    # Ensure chronological order (should already be, but just in case).
    df.sort_values('datetime', inplace=True)
    # Reset the index after sorting.
    df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------------------------
    # DATA INTEGRITY CHECKS
    # --------------------------------------------------------------------------

    # --- Check 1: Duplicate timestamps ---------------------------------------
    # If any datetime appears more than once, the data is corrupt.
    if df['datetime'].duplicated().any():
        dup_count = df['datetime'].duplicated().sum()
        print(f"[ERROR] Found {dup_count} duplicate timestamp(s). Data is corrupted.")
        return None

    # --- Check 2: Internal continuity (no gaps between first and last) -------
    # Generate a perfect hourly sequence from the minimum to maximum datetime.
    full_range = pd.date_range(
        start=df['datetime'].min(),
        end=df['datetime'].max(),
        freq='1h',
        tz='UTC'
    )
    # Find which expected hours are missing from the actual data.
    missing_hours = full_range.difference(df['datetime'])
    if len(missing_hours) > 0:
        print(f"[ERROR] Internal gaps detected: {len(missing_hours)} missing hour(s).")
        print(f"   First missing: {missing_hours[0]}")
        return None

    # --- Check 3: Realistic boundary conditions -------------------------------
    # Convert the requested start/end strings to timezone-aware UTC timestamps.
    expected_start_date = pd.to_datetime(start_str).tz_localize('UTC')
    expected_end_date = pd.to_datetime(end_str).tz_localize('UTC')

    # ---------- 3a. Start boundary ----------
    # We allow the first candle to be later than 00:00 on the start date because
    # many pairs began trading later that same day. However, it must not start
    # more than 23 hours after the requested start date.
    latest_acceptable_start = expected_start_date + pd.Timedelta(hours=23)
    if df['datetime'].min() > latest_acceptable_start:
        print(f"[ERROR] Data starts too late.")
        print(f"   Requested start date: {expected_start_date.date()}")
        print(f"   Actual start        : {df['datetime'].min()}")
        print(f"   Must start by       : {latest_acceptable_start}")
        return None

    # ---------- 3b. End boundary ----------
    # We treat "today" differently from historical end dates because we cannot
    # have future candles; the latest available data is up to the current hour.
    now_utc = pd.Timestamp.utcnow().floor('1h')   # Current UTC hour (e.g., 09:00)

    if expected_end_date.date() == now_utc.date():
        # The end date is today: the latest candle must be recent.
        # We allow a 2-hour lag to account for exchange API delays or slow
        # propagation of the most recent candle.
        earliest_acceptable_end = now_utc - pd.Timedelta(hours=2)
        if df['datetime'].max() < earliest_acceptable_end:
            print(f"[ERROR] Today's data is too stale.")
            print(f"   Latest candle : {df['datetime'].max()}")
            print(f"   Current UTC   : {now_utc}")
            return None
    else:
        # The end date is in the past: we expect data up to 23:00 of that day.
        expected_end_candle = expected_end_date + pd.Timedelta(hours=23)
        if df['datetime'].max() < expected_end_candle:
            print(f"[ERROR] Data does not cover full historical day.")
            print(f"   Expected last candle: {expected_end_candle}")
            print(f"   Actual last candle  : {df['datetime'].max()}")
            return None

    # --- Optional: Inform if data starts a bit later than 00:00 (normal) -----
    if df['datetime'].min() > expected_start_date:
        print(f"[INFO] Note: Data starts at {df['datetime'].min()}, not at 00:00. This is normal for newly listed pairs.")

    # If we reach this point, all checks have passed. Export the full contiguous
    # range; log-return and liquidity-cut scripts decide the final analysis window.
    print(
        f"[OK] Data validated over {len(df)} hours; exporting full range "
        f"from {df['datetime'].min()} to {df['datetime'].max()}."
    )
    return df


# ==============================================================================
# MAIN EXECUTION LOOP
# ==============================================================================
# Iterate over each trading pair and its earliest start date.
for symbol, start_date in PAIR_START_DATES.items():
    # If config download.from is set, it overrides per-pair listing date.
    effective_start = GLOBAL_FROM_STR if GLOBAL_FROM_STR else start_date
    # Construct a safe filename by removing the slash from the symbol.
    pair_name = symbol.replace('/', '')
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{pair_name}_{EXCHANGE_ID.upper()}_{TIMEFRAME}_complete.csv"
    )

    # Print a visual separator and information about the current pair.
    print("\n" + "=" * 60)
    print(f"Processing {symbol} -> start: {effective_start}, end: {END_DATE_STR}")
    print("=" * 60)

    # Call the download function.
    df_result = download_hourly_data(
        symbol=symbol,
        start_str=effective_start,
        end_str=END_DATE_STR,
        exchange_id=EXCHANGE_ID,
        timeframe=TIMEFRAME
    )

    # If a DataFrame is returned (i.e., validation passed), save it to CSV.
    if df_result is not None:
        # Add a human-readable string column for the datetime (optional).
        df_result['datetime_str'] = df_result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Select and order the columns we want in the output file.
        df_export = df_result[['datetime_str', 'open', 'high', 'low', 'close', 'volume']]
        # Write to CSV without an index column.
        df_export.to_csv(output_file, index=False, sep=',')
        print(f"Saved to: {output_file}")
    else:
        # Validation failed; an error message has already been printed.
        print(f"[ERROR] {symbol} - Failed to obtain complete and contiguous data.")

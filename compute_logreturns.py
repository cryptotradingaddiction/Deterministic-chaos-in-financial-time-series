#!/usr/bin/env python3
"""
Calculation of logarithmic differences (log-returns) from closing prices.

WHAT THIS PROGRAM DOES (DETAILED DESCRIPTION):
- The script is designed for automated batch processing of financial time series.
- It iterates through a predefined list of CSV files containing historical hourly data (candles).
- From the closing prices ('close' column), it mathematically calculates the logarithmic returns (log-returns) between two consecutive periods.
  Logarithmic returns are preferred in finance over simple percentage returns due to their additivity and time symmetry.
- The results are then exported into two different formats for each analyzed cryptocurrency:
    1. .dat file - "Raw" data. Contains only a single column of numerical values (one value per line).
       This format is ideal for quick import into statistical software (e.g., R, MATLAB) or for machine learning.
    2. _logreturns.csv - "Human-readable" data. Contains two columns: timestamp (datetime_str) and the calculated value (log_return).
       This format is suitable for checking in Excel or for plotting graphs where the time axis is important.

Usage in terminal / command line:
    python compute_logreturns.py
"""

# --- Import of necessary standard Python libraries ---
# These modules are part of the basic Python installation; no need to install anything via pip.

import os       # The 'os' (Operating System) library is used to interact with the OS.
                # Here we use it primarily for path manipulation (os.path) and safe file existence verification.
import csv      # The 'csv' library provides specialized tools for efficient reading and writing of comma-separated values.
                # It automatically handles issues with quotes or delimiters within text fields.
from math import log  # From the 'math' module, we import only the 'log' function.
from config_loader import load_config, get_data_dir
                      # In Python, the 'log' function without a base argument calculates the natural logarithm (base 'e'),
                      # which is exactly what is needed for calculating continuous (logarithmic) returns.

# Note: The original 'sys' import was removed as the script did not utilize any functions from it. This keeps the code clean.


# --- Main control function definition ---
def main():
    """
    Main entry point of the program.
    All application logic, loading, calculations, and saving are encapsulated in this function.
    Encapsulation in a function is a good practice that prevents polluting the global namespace.
    """
    config = load_config()
    INPUT_DIR = get_data_dir(config)

    # --- 1. Definition of the input data file list ---
    # We create a static list of strings.
    # Each string represents the relative path to one CSV file in the target directory.
    # These files must exist in the specified folder; otherwise, the program will report an error and skip them.
    files = [
        "BTCUSD_BITSTAMP_1h_complete.csv", 
        "ETHUSD_BITSTAMP_1h_complete.csv", 
        "LTCUSD_BITSTAMP_1h_complete.csv", 
        "XRPUSD_BITSTAMP_1h_complete.csv", 
        "LINKUSD_BITSTAMP_1h_complete.csv",
        "DOGEUSD_BITSTAMP_1h_complete.csv",
        "ADAUSD_BITSTAMP_1h_complete.csv",
    ]

    for filename in files:
        # Create the full path to the file (e.g., C:\DCh\data\BTCUSD...)
        input_file = os.path.join(INPUT_DIR, filename)
        
        print(f"\n==========================================")
        print(f"Starting to process: {input_file}")
        print(f"==========================================")

        # --- 2. Safety check for input file existence ---
        # The os.path.isfile() function returns True if the path points to an existing file.
        # The 'not' keyword reverses the logic. If the file does NOT exist, the indented block is executed.
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' does not exist on disk or is inaccessible. Skipping to the next cryptocurrency...")
            # The 'continue' keyword immediately ends the current loop iteration and moves to the next item in the list.
            # This ensures the program doesn't crash just because one file is missing.
            continue 

        # --- 3. Dynamic generation of names for output files ---
        # os.path.splitext() is ideal for safely handling extensions.
        # It splits a path like "data.csv" into a tuple: root ("data") and extension (".csv").
        base, ext = os.path.splitext(input_file)
        
        # Using the obtained "base", we create new filenames by appending 
        # custom text indicating what the file contains.
        dat_file = base + "_logreturns.dat" # Resulting name e.g.: "BTCUSD_BITSTAMP_1h_complete_logreturns.dat"
        csv_file = base + "_logreturns.csv" # Resulting name e.g.: "BTCUSD_BITSTAMP_1h_complete_logreturns.csv"

        # --- 4. Loading and cleaning source data (Extraction) ---
        print(f"Reading phase: Opening and loading data from file {input_file} ...")
        
        # Initialize an empty list 'data'. This list will serve as temporary storage in RAM.
        # We will store tuples containing only the two pieces of information we need: (time, price).
        data = []  

        # Open the input file using a "context manager" (the 'with' statement).
        # This is critical because 'with' guarantees the file closes automatically after reading (or upon error).
        # 'r' mode means "read". newline='' prevents issues with different line endings (Windows vs. Linux).
        # encoding='utf-8' ensures correct reading of special characters.
        with open(input_file, 'r', newline='', encoding='utf-8') as f:
            # Create a csv.reader instance to parse the opened file line by line.
            reader = csv.reader(f)
            
            # Use the next() function to explicitly read and "discard" the first line.
            # Most datasets have a text header (column names like "date,open,high,low,close,volume"),
            # which must not be included in mathematical calculations to avoid errors.
            header = next(reader)
            
            # The for loop iterates through all remaining lines in the file.
            # The 'row' variable is always a list of strings (individual cells).
            for row in reader:
                # Data integrity check: Ensure the row has a sufficient number of columns.
                # We assume index 4 (fifth column) contains the closing price.
                if len(row) < 6:
                    # Silently ignore and skip invalid rows.
                    continue  
                
                # Use a try-except block for safe data type conversion.
                # Data often contains empty values (""), "NaN", or non-numeric characters.
                try:
                    # Index 0 usually contains the timestamp. .strip() removes whitespace.
                    dt_str = row[0].strip()           
                    # Index 4 extracts the close price. float() converts the string to a decimal number.
                    close = float(row[4])             
                    
                    # If conversion to float succeeds, append the (time, price) tuple to our list.
                    data.append((dt_str, close))
                
                # If float() encounters text that cannot be converted (e.g., an empty string), it raises a ValueError.
                except ValueError:
                    # Catch the exception, print a warning with the faulty row content, and continue.
                    print(f"Warning: Skipping row with invalid or non-numeric data: {row}")
                    continue

        # --- 5. Check for minimum required amount of data ---
        # To calculate a difference, we logically need at least two values (past and present).
        if len(data) < 2:
            print("Error: Insufficient data for mathematical difference calculation (at least 2 valid rows needed). Skipping this file...")
            continue 

        print(f"Successfully loaded {len(data)} valid records.")
        print("Calculation phase: Applying natural logarithm and calculating differences...")

        # --- 6. Main analytical core: Log-return calculation ---
        # Initialize an empty list to store the final calculated returns.
        log_returns = []  

        # Use a loop with the range() function.
        # We start from index 1 (the second element), not 0.
        # This is because we need to look "one step back" (index i-1).
        # The first row (index 0) has no predecessor, so no return can be calculated for it.
        for i in range(1, len(data)):
            # data[i-1] returns the tuple of the previous row: (time, price). [1] selects the price.
            prev_close = data[i-1][1]   # Closing price at time t-1 (previous hour)
            curr_close = data[i][1]     # Closing price at time t (current hour)

            # Mathematical domain handling: Logarithm is not defined for zero or negative numbers.
            # Although asset prices shouldn't be negative, it can happen in bad data (API errors, flash-crashes).
            if prev_close <= 0 or curr_close <= 0:
                print(f"Warning: Anomalous (zero or negative) price found at row {i}. Log calculation would fail, skipped.")
                continue 

            # The equation for logarithmic return calculation.
            # Mathematically corresponds to: R = ln(P_t) - ln(P_{t-1}), which is equivalent to ln(P_t / P_{t-1}).
            ret = log(curr_close) - log(prev_close)
            
            # Store the calculated value in the list.
            log_returns.append(ret)

        # --- 7. Saving the first set of results: Raw .dat format ---
        # Open the 'dat_file' for writing ('w' mode). If it exists, it will be overwritten.
        with open(dat_file, 'w') as f:
            for val in log_returns:
                # Write the number as text.
                # {val:.10f} formats the float to exactly 10 decimal places.
                # This prevents scientific notation (e.g., 1.2e-5) which might confuse other software.
                # The \n ensures each number is on its own line.
                f.write(f"{val:.10f}\n")
        
        print(f"Writing phase: Saved {len(log_returns)} raw numerical values to {dat_file}")

        # --- 8. Saving the second set of results: Structured .CSV format ---
        # Open the 'csv_file' for writing. Use utf-8 and prevent double line endings.
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write the table header so the user knows what the columns represent in Excel.
            writer.writerow(["datetime_str", "log_return"])
            
            # Use enumerate() to get both the value (ret) and its index (i).
            for i, ret in enumerate(log_returns):
                # We must pair the calculated return with the correct timestamp.
                # Since the first return (index 0 in log_returns) was calculated using indices 0 and 1
                # from the original 'data', it relates to the time at index 1.
                datetime_str = data[i+1][0]
                
                # Write the paired row: [time, formatted return value].
                writer.writerow([datetime_str, f"{ret:.10f}"])
                
        print(f"Writing phase: Saved {len(log_returns)} structured records (time + value) to {csv_file}")

# --- Execution protection (Standard Python convention) ---
# This condition checks if the script is being run directly by the user or imported as a module.
# If run directly, __name__ equals "__main__" and our main() function is called.
if __name__ == "__main__":
    main()

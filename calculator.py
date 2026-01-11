import math
import csv
import sys
from pathlib import Path


def compute_forward_factor(dte1, iv1, dte2, iv2):
    """
        dte1: Days to expiry for front month
        iv1: Implied vol for front month (%)
        dte2: Days to expiry for back month
        iv2: Implied vol for back month (%)
    
    result:
        fwd_vol_pct, ff_ratio, ff_pct, error_msg
    """

    if dte1 < 0 or dte2 < 0:
        return None, None, None, "negative"
    if dte2 <= dte1:
        return None, None, None, "DTE2 must be > DTE1"
    if iv1 < 0 or iv2 < 0:
        return None, None, None, "ivs negative"
    
    # annualized terms
    T1 = dte1 / 365.0
    T2 = dte2 / 365.0
    s1 = iv1 / 100.0
    s2 = iv2 / 100.0
    
    # Total variances
    tv1 = (s1 ** 2) * T1
    tv2 = (s2 ** 2) * T2
    
    # Forward variance
    denom = T2 - T1
    fwd_var = (tv2 - tv1) / denom
    
    if fwd_var < 0:
        return None, None, None, "neg forward variance"
    
    # Forward volatility 
    fwd_sigma = math.sqrt(fwd_var)
    fwd_vol_pct = fwd_sigma * 100
    
    # Forward Factor
    if fwd_sigma == 0.0:
        return fwd_vol_pct, None, None, "Forward vol is zero"
    
    ff_ratio = (s1 - fwd_sigma) / fwd_sigma
    ff_pct = ff_ratio * 100
    
    return fwd_vol_pct, ff_ratio, ff_pct, None


def process_csv(filepath, dte1=30, dte2=60):

    results = []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            if not all(col in reader.fieldnames for col in ['ticker', 'iv1', 'iv2']):
                print("error missing columns")
                sys.exit(1)
            
            for row in reader:
                ticker = row['ticker'].strip()
                
                try:
                    iv1 = float(row['iv1'])
                    iv2 = float(row['iv2'])
                except ValueError:
                    results.append((ticker, None, None, "invalid IV "))
                    continue
                
                fwd_vol, ff_ratio, ff_pct, error = compute_forward_factor(
                    dte1, iv1, dte2, iv2
                )
                
                results.append((ticker, ff_pct, fwd_vol, error))
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    return results


def print_results(results):
    """ results descending """

    valid = [(t, ff, fv) for t, ff, fv, err in results if ff is not None]
    invalid = [(t, err) for t, ff, fv, err in results if ff is None]

    valid.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 80)
    print("FORWARD FACTOR RESULTS (sorted by FF% descending)")
    print("=" * 80)
    print(f"{'Ticker':<10} {'FF%':>12} {'Fwd Vol%':>12}")
    print("-" * 80)
    
    for ticker, ff_pct, fwd_vol in valid:
        print(f"{ticker:<10} {ff_pct:>12.4f} {fwd_vol:>12.4f}")
    
    if invalid:
        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)
        for ticker, error in invalid:
            print(f"{ticker:<10} - {error}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(valid)} valid, {len(invalid)} errors")
    print("=" * 80 + "\n")


def main():

    CSV_FILE = "finaliv.csv" 
    DTE1 = 30  
    DTE2 = 60 

    print(f"CSV: {CSV_FILE}")
    
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            print(f"Columns found: {fieldnames}")
            print("-" * 80)
            
            rows = []
            for i, row in enumerate(reader):
                if i < 5:  
                    print(row)
                rows.append(row)
            
            print(f"\nTotal rows: {len(rows)}")
            print("=" * 80)
    
    except FileNotFoundError:
        print("file not found")
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    print(f"\n DTE1 = {DTE1} days, DTE2 = {DTE2} days")
    results = process_csv(CSV_FILE, DTE1, DTE2)
    print_results(results)


if __name__ == "__main__":
    main()
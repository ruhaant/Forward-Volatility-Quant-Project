import math
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Option
import time

from ib_insync import IB
import pandas as pd
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=101)

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# Black-Scholes price 
def _bs_price(spot, strike, t, r, q, vol, option_type='C'):
    if t <= 0 or vol <= 0:
        #  bad cases
        if option_type == 'C':
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if option_type == 'C':
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    else:
        return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)

#  Newton-Raphson iv
def _implied_vol_from_price(mkt_price, spot, strike, t, r=0.01, q=0.0, option_type='C', initial_vol=0.3):
    if mkt_price <= 0:
        raise ValueError("negative market price")
    vol = initial_vol
    for i in range(60):
        price = _bs_price(spot, strike, t, r, q, vol, option_type)
        # Vega approx
        if vol <= 0:
            vol = 1e-6
        d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
        vega = spot * math.exp(-q * t) * math.sqrt(t) * (1.0 / math.sqrt(2*math.pi)) * math.exp(-0.5 * d1 * d1)
        diff = price - mkt_price
        if abs(diff) < 1e-8:
            return vol
        # Newton step
        step = diff / vega if vega > 1e-12 else diff / (1e-6)
        vol -= step
        # keep vol in reasonable bounds
        if vol <= 1e-8:
            vol = 1e-8
        if vol > 5.0:
            vol = 5.0
    # if no convergence return last value
    return vol

def get_atm_call_iv(ib: IB,
                    symbol: str,
                    target_dte: int = 30,
                    expiry_iso: str = None,
                    exchanges_to_try: list = None,
                    strike_side_window: int = 1,
                    snapshot_wait_s: float = 1.2,
                    r: float = 0.02,
                    q: float = 0.0,
                    verbose: bool = False) -> tuple:

    if exchanges_to_try is None:
        exchanges_to_try = ['SMART', 'CBOE']


    stock = Stock(symbol, "SMART", "USD")
    try:
        ib.qualifyContracts(stock)
    except Exception as e:
        raise ValueError(f"qualify problem this: {symbol}: {e}")


    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        raise ValueError(f"no chains found for {symbol}")

    chain = max(chains, key=lambda c: len(getattr(c, "expirations", [])))
    expirations = sorted(chain.expirations)
    strikes = sorted(chain.strikes)
    chain_exchange = getattr(chain, "exchange", None)
    trading_class = getattr(chain, "tradingClass", symbol)

    if verbose:
        print(f"chain exchange={chain_exchange} tradingClass={trading_class}")
        print(f"expirations sample: {expirations[:6]}")
        print(f"strikes count: {len(strikes)} (sample: {strikes[:6]})")


    if expiry_iso is None:
        today = datetime.now().date()
        exp_candidates = []
        for e in expirations:
            try:
                ddate = datetime.strptime(e, "%Y%m%d").date()
            except Exception:
                continue
            dte = (ddate - today).days
            if dte > 0:
                exp_candidates.append((dte, e, ddate))
        if not exp_candidates:
            raise ValueError(f"no future expirations available for {symbol}")
        exp_choice = min(exp_candidates, key=lambda x: abs(x[0] - target_dte))
        dte = exp_choice[0]
        expiry_iso = exp_choice[1]
    else:
        if expiry_iso not in expirations:
            raise ValueError(f"Expiry {expiry_iso} not present for {symbol}")
        dte = (datetime.strptime(expiry_iso, "%Y%m%d").date() - datetime.now().date()).days
        if dte <= 0:
            raise ValueError(f"Expiry {expiry_iso} not in the future")


    spot_price = None
    try:
        bars = ib.reqHistoricalData(stock, endDateTime='', durationStr='1 D', barSizeSetting='1 day',
                                    whatToShow='TRADES', useRTH=True, formatDate=1)
        if bars and len(bars) > 0:
            spot_price = bars[-1].close
    except Exception:
        pass
    if not spot_price or spot_price <= 0:
        if not strikes:
            raise ValueError("No spot and no strikes to estimate ATM")
        spot_price = strikes[len(strikes) // 2]


    if not strikes:
        raise ValueError("No strikes in chain")
    atm_strike = min(strikes, key=lambda s: abs(s - spot_price))
    idx = strikes.index(atm_strike)
    start = max(0, idx - strike_side_window)
    end = min(len(strikes), idx + strike_side_window + 1)
    candidate_strikes = strikes[start:end]

    if verbose:
        print(f"chosen expiry {expiry_iso} ({dte} DTE). ATM strike {atm_strike}. strikes {candidate_strikes}")


    exp_date = datetime.strptime(expiry_iso, "%Y%m%d").date()
    t_years = max((exp_date - datetime.now().date()).days / 365.0, 1e-6)
    collected = []
    used_strike = None

    for strike in candidate_strikes:

        call_contract = None
        for exch in exchanges_to_try:
            try:
                call = Option(symbol, expiry_iso, strike, 'C', exch, tradingClass=trading_class, multiplier='100')
                qual = None
                try:
                    qual = ib.qualifyContracts(call)
                except Exception:
                    qual = []
                if qual and len(qual) > 0:
                    call_contract = qual[0]
                    if verbose:
                        print(f"strike {strike} qualified on {exch} as conId={call_contract.conId}")
                    break
            except Exception:
                continue

        if not call_contract:
            if verbose:
                print(f"strike {strike} did not qualify on preferred exchanges {exchanges_to_try}")
            continue


        tkr = ib.reqMktData(call_contract, "", snapshot=True)
        if snapshot_wait_s > 0:
            ib.sleep(snapshot_wait_s)


        mg = getattr(tkr, "modelGreeks", None)
        if mg and getattr(mg, "impliedVol", None) and mg.impliedVol > 0:
            iv = float(mg.impliedVol) * 100.0
            collected.append(iv)
            used_strike = strike
            if verbose:
                print(f"strike {strike}: modelGreeks IV {iv:.4f}%")
            break


        lg = getattr(tkr, "lastGreeks", None)
        if lg and getattr(lg, "impliedVol", None) and lg.impliedVol > 0:
            iv = float(lg.impliedVol) * 100.0
            collected.append(iv)
            used_strike = strike
            if verbose:
                print(f"strike {strike}: lastGreeks IV {iv:.4f}%")
            break


        b = getattr(tkr, "bid", None)
        a = getattr(tkr, "ask", None)
        l = getattr(tkr, "last", None)
        mark = getattr(tkr, "mark", None)
        mid = None
        if b and a and b > 0 and a > 0:
            mid = (b + a) / 2.0
        elif l and l > 0:
            mid = l
        elif mark and mark > 0:
            mid = mark

        if mid and mid > 0:
            try:
                iv_dec = _implied_vol_from_price(mid, spot_price, strike, t_years, r=r, q=q, option_type='C')
                iv_pct = iv_dec * 100.0
                collected.append(iv_pct)
                used_strike = strike
                if verbose:
                    print(f"strike {strike}: mid {mid:.4f} -> IV {iv_pct:.4f}%")
                break
            except Exception as e:
                if verbose:
                    print(f"strike {strike}: BS fallback failed: {e}")


        if verbose:
            print(f"strike {strike}: no usable data (model, last, or mid)")

    if not collected:
        raise ValueError(f"No IV found for {symbol} expiry {expiry_iso}. Tried strikes: {candidate_strikes} on exchanges {exchanges_to_try}")

    iv_avg = float(sum(collected) / len(collected))
    return iv_avg, expiry_iso, dte, used_strike or atm_strike, spot_price

# iv_pct, expiry, dte, strike_used, spot = get_atm_call_iv(ib, "AAPL", target_dte=60, verbose=True)
# print(iv_pct, expiry, dte, strike_used, spot)
tickers = ["MSFT", "GOOGL", "META", "VZ", "LLY", "MA", "MRK"]
    # "NVDA", "AMD", "QCOM", "INTC", "AVGO", "MU", "TSLA", "F", "GM",
    # "TM", "NIO", "RGTI", "RDDT", "BABA", "CRML", "PDD", "AMZN", "CAT",
    # "DE", "BA", "XLE", "SMCI", "PLTR", "MRNA", "VRTX", "XBI"


ib = IB()
try:
    # connect once
    print("Connecting to IB...")
    #API stuff here



    results = []

    for t in tickers:
        print(f"\nProcessing {t} ...")
        try:

            iv1, expiry1, dte1, strike1, spot1 = get_atm_call_iv(ib, t, target_dte=30, verbose=False)

            time.sleep(0.5)
            iv2, expiry2, dte2, strike2, spot2 = get_atm_call_iv(ib, t, target_dte=60, verbose=False)
            row = {
                "ticker": t,
                "iv1_pct": iv1,
                "expiry1": expiry1,
                # "dte1": dte1,
                "strike1": strike1,
                # "spot1": spot1,
                "iv2_pct": iv2,
                "expiry2": expiry2,
                # "dte2": dte2,
                "strike2": strike2,
                # "spot2": spot2,
                "status": "ok"
            }
            print(f"  {t}: IV30={iv1:.2f}%, IV60={iv2:.2f}%")
        except Exception as e:
            # log error and continue
            row = {
                "ticker": t,
                "iv1_pct": None,
                "expiry1": None,
                "dte1": None,
                "strike1": None,
                "spot1": None,
                "iv2_pct": None,
                "expiry2": None,
                "dte2": None,
                "strike2": None,
                "spot2": None,
                "status": f"error: {e}"
            }
            print(f"  Error fetching {t}: {e}")
        results.append(row)


        time.sleep(2.0)

    #  save
    df = pd.DataFrame(results)
    df.to_csv("finaliv.csv", index=False)
    print("\nWrote finaliv.csv")

finally:
    if ib.isConnected():
        print("Disconnecting...")
        ib.disconnect()

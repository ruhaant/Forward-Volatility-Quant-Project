import math
from datetime import datetime, timedelta
from ib_insync import IB, Stock, Option

from ib_insync import IB
import pandas as pd
ib = IB()

#api connect here

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_price(spot, strike, t, r, q, vol, option_type='C'):
    if t <= 0 or vol <= 0:
        return max(0.0, spot - strike) if option_type == 'C' else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if option_type == 'C':
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    else:
        return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)

def _implied_vol_from_price(mkt_price, spot, strike, t, r=0.02, q=0.0, option_type='C', initial_vol=0.3):
    if mkt_price <= 0:
        raise ValueError("Market price must be positive to compute IV")
    vol = initial_vol
    for _ in range(80):
        price = _bs_price(spot, strike, t, r, q, vol, option_type)
        # compute vega (approx)
        if vol <= 0:
            vol = 1e-6
        d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
        vega = spot * math.exp(-q * t) * math.sqrt(t) * (1.0 / math.sqrt(2*math.pi)) * math.exp(-0.5 * d1 * d1)
        diff = price - mkt_price
        if abs(diff) < 1e-8:
            return vol
        step = diff / (vega if vega > 1e-12 else 1e-6)
        vol -= step
        # keep vol in bounds
        if vol <= 1e-8:
            vol = 1e-8
        if vol > 5.0:
            vol = 5.0
    return vol

def get_atm_call_iv(ib: IB,
                    symbol: str,
                    target_dte: int = 30,
                    expiry_iso: str = None,
                    prefer_exchange: str = "SMART",
                    max_window_strikes: int = 3,
                    snapshot_wait_s: float = 1.5,
                    r: float = 0.02,
                    q: float = 0.0,
                    verbose: bool = False) -> float:

    # qualify 
    stock = Stock(symbol, "SMART", "USD")
    try:
        ib.qualifyContracts(stock)
    except Exception as e:
        raise ValueError(f"Failed to qualify underlying {symbol}: {e}")

    # pull option chains 
    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        raise ValueError(f"No option chains found for {symbol}")

    # chain with most expirations 
    chosen_chain = max(chains, key=lambda c: len(getattr(c, "expirations", [])))
    expirations = sorted(chosen_chain.expirations)
    strikes = sorted(chosen_chain.strikes)
    chain_exchange = getattr(chosen_chain, "exchange", None)
    trading_class = getattr(chosen_chain, "tradingClass", symbol)

    if verbose:
        print(f"chain exchange={chain_exchange} tradingClass={trading_class}")
        print(f"available expirations (sample): {expirations[:8]}")
        print(f"strike count: {len(strikes)} (sample: {strikes[:8]})")

    # expiry selection
    if expiry_iso is None:
        exp_candidates = []
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y%m%d").date()
            except Exception:
                continue
            dte = (exp_date - today).days
            if dte > 0:
                exp_candidates.append((dte, exp, exp_date))
        if not exp_candidates:
            raise ValueError(f"No future expirations available for {symbol}")
        # nearest to target_dte
        exp_choice = min(exp_candidates, key=lambda x: abs(x[0] - target_dte))
        dte = exp_choice[0]
        expiry_iso = exp_choice[1]
    else:
        if expiry_iso not in expirations:
            raise ValueError(f"Requested expiry {expiry_iso} not in chain for {symbol}")
        # compute DTE
        dte = (datetime.strptime(expiry_iso, "%Y%m%d").date() - datetime.now().date()).days
        if dte <= 0:
            raise ValueError(f"Expiry {expiry_iso} is not in the future")

    # determine a spot/estimate
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
            raise ValueError("No spot and no strikes to estimate an ATM")
        spot_price = strikes[len(strikes) // 2]  # fallback estimate

    # pick ATM strike 
    if not strikes:
        raise ValueError("No strikes available in chain")
    atm_strike = min(strikes, key=lambda s: abs(s - spot_price))
    idx = strikes.index(atm_strike)
    # neighborhood of strikes by index (robust)
    start = max(0, idx - max_window_strikes)
    end = min(len(strikes), idx + max_window_strikes + 1)
    candidate_strikes = strikes[start:end]

    if verbose:
        print(f"Selected expiry {expiry_iso} ({dte} DTE), ATM strike from chain = {atm_strike}")
        print(f"Candidate strikes (by index): {candidate_strikes}")

    collected_ivs = []
    exp_date = datetime.strptime(expiry_iso, "%Y%m%d").date()
    t_years = max((exp_date - datetime.now().date()).days / 365.0, 1e-6)

    for strike in candidate_strikes:
        exchanges_to_try = [prefer_exchange]
        if chain_exchange and chain_exchange not in exchanges_to_try:
            exchanges_to_try.append(chain_exchange)
        qualified = None
        call_contract = None

        for exch in exchanges_to_try:
            try:
                call = Option(symbol, expiry_iso, strike, 'C', exch, multiplier='100')
                # qualify
                qual = None
                try:
                    qual = ib.qualifyContracts(call)
                except Exception:
                    qual = []
                if qual and len(qual) > 0:
                    call_contract = qual[0]
                    break
            except Exception:
                continue

        if not call_contract:
            if verbose:
                print(f"strike {strike} did not qualify on tried exchanges {exchanges_to_try}")
            continue

        # request market data snapshot slow
        tkr = ib.reqMktData(call_contract, "", snapshot=True)
        if snapshot_wait_s > 0:
            ib.sleep(snapshot_wait_s)

        # modelGreeks.impliedVol (decimal)
        iv_from_model = None
        mg = getattr(tkr, "modelGreeks", None)
        if mg and getattr(mg, "impliedVol", None):
            try:
                iv_from_model = float(mg.impliedVol) * 100.0
            except Exception:
                iv_from_model = None

        if iv_from_model is not None and iv_from_model > 0:
            collected_ivs.append(iv_from_model)
            if verbose:
                print(f"strike {strike}: got model implied vol {iv_from_model:.4f}%")
            # no midprice needed
            continue

        # method 2 lastGreeks.impliedVol
        lg = getattr(tkr, "lastGreeks", None)
        iv_from_lastgreek = None
        if lg and getattr(lg, "impliedVol", None):
            try:
                iv_from_lastgreek = float(lg.impliedVol) * 100.0
            except Exception:
                iv_from_lastgreek = None
        if iv_from_lastgreek is not None and iv_from_lastgreek > 0:
            collected_ivs.append(iv_from_lastgreek)
            if verbose:
                print(f"strike {strike}: got lastGreeks implied vol {iv_from_lastgreek:.4f}%")
            continue

        # method 3: mid price (bid/ask/last/mark) 
        def _mid(md):
            b = getattr(md, "bid", None)
            a = getattr(md, "ask", None)
            l = getattr(md, "last", None)
            mark = getattr(md, "mark", None)
            if b and a and b > 0 and a > 0:
                return (b + a) / 2.0
            if l and l > 0:
                return l
            if mark and mark > 0:
                return mark
            return None

        mid_price = _mid(tkr)
        if mid_price and mid_price > 0:
            # compute implied vol from mid price (call)
            try:
                iv_dec = _implied_vol_from_price(mid_price, spot_price, strike, t_years, r=r, q=q, option_type='C')
                iv_pct = iv_dec * 100.0
                collected_ivs.append(iv_pct)
                if verbose:
                    print(f"strike {strike}: computed IV from mid price {mid_price:.4f} -> {iv_pct:.4f}%")
                continue
            except Exception as e:
                if verbose:
                    print(f"strike {strike}: BS fallback failed: {e}")



        if verbose:
            print(f"strike {strike}: no modelGreeks, lastGreeks, or mid price available")

    if not collected_ivs:
        raise ValueError(f"No IV data available for {symbol} expiry {expiry_iso} (tried strikes: {candidate_strikes})"
                         " - check OPRA / market-data permissions or try different expiry.")

    # average IVs collected across strikes (percent)
    iv_avg = float(sum(collected_ivs) / len(collected_ivs))
    return iv_avg

#MAIN LIST
tickers=["MSFT", "GOOGL", "META", "VZ", "LLY", "MA", "MRK"] 
        #  "TM", "NIO", "LI", "XPEV", "BABA", "BIDU", "PDD", "AMZN", "CAT", 
        #  "DE", "BA", "XLE", "SMCI", "PLTR", "MRNA", "VRTX", "XBI"]

result = pd.DataFrame(columns=["Ticker", "IV1","IV2"])
for i in range(len(tickers)):
    ticker=tickers[i]
    iv1 = get_atm_call_iv(ib, ticker, target_dte=30, verbose=False)
    iv2 = get_atm_call_iv(ib, ticker, target_dte=30, verbose=False)
    result.loc[i]=[ticker,iv1,iv2]

result.to_csv("iv.csv")

ib.disconnect()
"""
Puxa preço real dos ETFs da B3 via yfinance (sufixo .SA). Grátis, sem chave.

Cobertura confirmada: BOVA11, ISUS11, GOVE11 têm histórico bom no Yahoo Finance.
REVE11 é um ETF mais novo (temático de transição energética) — pode ter menos
histórico ou spread maior de bid/ask nos dados; se vier vazio ou muito curto,
considere puxar direto da corretora pra esse ticker específico.

Uso:
    python fetch_prices_yfinance.py
    python fetch_prices_yfinance.py --years 3

Salva em data/raw/etf_prices.csv (formato longo: data, ticker, close).
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import yfinance as yf

from momentum_punch import config


def fetch_prices(tickers: list[str] = config.TICKERS, years: int = 3) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    frames = []
    for ticker in tickers:
        yf_ticker = f"{ticker}.SA"
        try:
            data = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty:
                print(f"[fetch_prices] {yf_ticker}: nenhum dado retornado (ticker pode não existir no Yahoo Finance ou ter pouco histórico)")
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            df = close.reset_index()
            df.columns = ["data", "close"]
            df["ticker"] = ticker
            frames.append(df)
            print(f"[fetch_prices] {yf_ticker}: {len(df)} dias de preço")
        except Exception as exc:
            print(f"[fetch_prices] Falha em {yf_ticker}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["data", "ticker", "close"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--out", default="data/raw/etf_prices.csv")
    args = parser.parse_args()

    df = fetch_prices(years=args.years)
    df.to_csv(args.out, index=False)
    print(f"\n[fetch_prices] Salvo em {args.out} ({len(df)} linhas, {df['ticker'].nunique() if not df.empty else 0} tickers)")

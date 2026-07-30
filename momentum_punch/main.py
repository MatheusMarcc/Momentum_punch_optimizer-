"""
Demo end-to-end do Momentum Punch usando dados SINTÉTICOS (preços + sentiment).

Uso:
    python main.py

Pra rodar com dados reais:
  1. Substitua momentum_punch.synthetic_data.generate_prices() por cotações
     reais da B3 (ex: exportadas da sua corretora, MetaTrader5, ou provedor
     pago tipo Cedro/Comdinheiro — não uso Yahoo Finance porque a cobertura
     de ETFs da B3 lá é ruim).
  2. Substitua generate_sentiment_scores()/generate_stress_index() pelo
     resultado real de momentum_punch.sentiment.score_history(), alimentado
     pelos textos vindos de momentum_punch.data_collection.
  3. Rode run_backtest() do mesmo jeito.
"""
from momentum_punch import backtest, config, synthetic_data


def main():
    prices = synthetic_data.generate_prices()
    cdi = synthetic_data.generate_cdi_daily_return(n_days=len(prices))
    sentiment = synthetic_data.generate_sentiment_scores(prices.index)
    stress = synthetic_data.generate_stress_index(prices.index)

    result = backtest.run_backtest(
        prices=prices,
        cdi_daily_return=cdi,
        sentiment_scores=sentiment,
        stress_index=stress,
        benchmark="60_40",
    )

    print("\n=== Métricas de performance ===")
    for strategy, metrics in result["metrics"].items():
        print(f"\n{strategy}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    n_riskoff = sum(
        1 for w in result["weights_history"].values() if w.get("_mode") == "RISK-OFF"
    )
    print(f"\nRebalanceamentos em modo RISK-OFF: {n_riskoff} / {len(result['weights_history'])}")

    last_date = max(result["weights_history"])
    print(f"\nÚltima alocação ({last_date.date()}):")
    for k, v in result["weights_history"][last_date].items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()

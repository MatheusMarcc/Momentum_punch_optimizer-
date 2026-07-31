"""Engineering demonstration using explicitly synthetic inputs."""
from momentum_punch import backtest, synthetic_data


def main() -> None:
    prices = synthetic_data.generate_prices()
    cdi = synthetic_data.generate_cdi_daily_return(
        n_days=len(prices), dates=prices.index
    )
    sentiment = synthetic_data.generate_sentiment_scores(prices.index)
    stress = synthetic_data.generate_stress_index(prices.index)

    result = backtest.run_backtest(
        prices=prices,
        cdi_daily_return=cdi,
        sentiment_scores=sentiment,
        stress_index=stress,
        benchmark="60_40",
    )

    print("\nTESTE DE ENGENHARIA — DADOS SINTÉTICOS")
    print("Não usar estes números como evidência de desempenho financeiro.\n")
    for strategy, metrics in result["metrics"].items():
        print(strategy)
        for key, value in backtest.format_metrics(metrics).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

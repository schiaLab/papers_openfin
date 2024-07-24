import numpy as np
import pandas as pd


def rotation_strategy_eval(bench: pd.Series, results: dict[str, np.ndarray], days_per_year: int) -> pd.DataFrame:
    """
    주어진 벤치마크와 회전 전략 결과를 기반으로 여러 평가 지표를 계산합니다.

    매개변수:
    bench (pd.Series): 벤치마크 수익률 시계열.
    results (dict[str, np.ndarray]): 전략 이름을 키로 하고 회전 신호의 numpy 배열을 값으로 하는 딕셔너리.
    days_per_year: 전략 신호가 1년에 몇 번 있는지에 대한 정수.
    반환값:
    pd.DataFrame: 각 전략에 대한 평가 지표를 포함한 데이터프레임.
    """
    evaluation = {}

    for ind, x in results.items():
        # 회전 전략을 기반으로 한 수익률 계산
        strategy_returns = bench * x

        # 포지션 변경 빈도 계산 (거래 빈도)
        turnover = pd.Series(np.where(x > 0, 1, 0)).diff().abs().fillna(0)

        # 누적 총 수익률 계산
        cumulative_gross_returns = strategy_returns.sum() - (-bench.sum())

        # 거래 횟수 추정
        number_of_trades = np.sum(turnover)

        # 수익률 0을 위한 거래 비용 계산 (손익분기 거래 비용)
        if number_of_trades > 0:
            break_even_transactional_cost_per_trade = cumulative_gross_returns / number_of_trades
        else:
            break_even_transactional_cost_per_trade = np.nan  # 거래가 없을 경우 정의되지 않음

        # 거래 비용을 고려한 수익률 계산
        strategy_returns100 = bench * x - (0.01 * turnover.values)
        strategy_returns50 = bench * x - (0.005 * turnover.values)
        strategy_returns25 = bench * x - (0.0025 * turnover.values)

        # 여러 지표 계산
        annualized_geo_mean_return = np.prod(1 + strategy_returns) ** (days_per_year / len(strategy_returns)) - 1
        annualized_geo_mean_return100 = np.prod(1 + strategy_returns100) ** (days_per_year / len(strategy_returns)) - 1
        annualized_geo_mean_return50 = np.prod(1 + strategy_returns50) ** (days_per_year / len(strategy_returns)) - 1
        annualized_geo_mean_return25 = np.prod(1 + strategy_returns25) ** (days_per_year / len(strategy_returns)) - 1
        annualized_arith_mean_return = strategy_returns.mean() * days_per_year
        annualized_median_return = (1 + strategy_returns.median()) ** days_per_year - 1
        annualized_std_dev = strategy_returns.std() * np.sqrt(days_per_year)
        return_to_risk_ratio = annualized_arith_mean_return / annualized_std_dev

        # Hit ratio 및 평균 수익률 계산
        hits = strategy_returns[strategy_returns > 0]
        misses = strategy_returns[strategy_returns <= 0]
        hit_ratio = len(hits) / len(strategy_returns)
        avg_return_correct = hits.mean()
        avg_return_wrong = misses.mean()

        # 결과를 데이터프레임으로 정리
        evaluation[ind] = pd.Series({
            'Annualized Geometric Mean Return': annualized_geo_mean_return,
            'Annualized Geometric Mean Return (25bp Transaction Cost)': annualized_geo_mean_return25,
            'Annualized Geometric Mean Return (50bp Transaction Cost)': annualized_geo_mean_return50,
            'Annualized Arithmetic Mean Return (100bp Transaction Cost)': annualized_geo_mean_return100,
            'Annualized Arithmetic Mean Return': annualized_arith_mean_return,
            'Break-Even Transactional Cost For Excess Return': break_even_transactional_cost_per_trade,
            'Annualized Median Return': annualized_median_return,
            'Annualized Standard Deviation': annualized_std_dev,
            'Return-to-Risk Ratio': return_to_risk_ratio,
            'Hit Ratio': hit_ratio,
            'Average Return When Correct': avg_return_correct,
            'Average Return When Wrong': avg_return_wrong
        })

    return pd.DataFrame(evaluation).round(4)

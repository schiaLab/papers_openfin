import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

import copy

def winsor(df0: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    데이터프레임에 윈저화(winsorization)를 적용합니다. 윈저화는 데이터의 극단적인 값을 제한하여 이상치로 인한 영향을 줄입니다.

    매개변수:
    df0 (pd.DataFrame): 윈저화를 적용할 입력 데이터프레임.
    threshold (float): 윈저화에 사용할 분위수 임계값. 이 임계값으로 지정된 분위수 외부의 값들은 해당 임계값으로 대체됩니다.

    반환값:
    pd.DataFrame: 윈저화된 값을 가진 데이터프레임.
    """

    def winsor2(series: pd.Series, threshold: float) -> pd.Series:
        """
        단일 판다스 시리즈에 윈저화를 적용합니다.

        매개변수:
        series (pd.Series): 윈저화를 적용할 입력 시리즈.
        threshold (float): 윈저화에 사용할 분위수 임계값.

        반환값:
        pd.Series: 윈저화된 값을 가진 시리즈.
        """
        series0 = series.copy()  # 원본 데이터를 수정하지 않기 위해 시리즈를 복사합니다.
        over = series0.dropna().quantile(1 - threshold)  # 상위 분위수를 계산합니다.
        under = series0.dropna().quantile(threshold)  # 하위 분위수를 계산합니다.

        # 상위 분위수 값을 초과하는 값을 상위 분위수 값으로 제한합니다.
        series0 = series0.where(series0 < over, over)
        # 하위 분위수 값 이하의 값을 하위 분위수 값으로 제한합니다.
        series0 = series0.where(series0 > under, under)

        return series0

    df = df0.copy()  # 원본 데이터를 수정하지 않기 위해 데이터프레임을 복사합니다.
    # 데이터프레임의 각 열에 윈저화를 적용합니다.
    df = df.apply(winsor2, axis=1, threshold=threshold)
    # 원본 데이터프레임의 NaN 값을 보존합니다.
    df[df0.isna()] = np.nan

    return df




def resizer_month(df00: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    월별로 업데이트되지 않지만 연간보다는 자주 업데이트되는 데이터를 월간 데이터프레임에 채웁니다.

    매개변수:
    df00 (pd.DataFrame): 원본 데이터가 있는 데이터프레임, 월별로 업데이트되지 않지만 연간보다는 자주 업데이트됩니다.
    reference (pd.DataFrame): 월간 빈도로 다시 색인을 지정할 참조 데이터프레임.

    반환값:
    pd.DataFrame: 원본 데이터를 채운 후 월간 참조 데이터프레임에 맞게 정렬된 데이터프레임.
    """
    if not isinstance(df00.index, pd.DatetimeIndex) or not isinstance(reference.index, pd.DatetimeIndex):
        raise ValueError("입력하는 데이터프레임들의 인덱스는 Datetime index여야만 합니다.")

    # 원본 데이터프레임을 깊은 복사하여 수정하지 않도록 합니다.
    df0 = copy.deepcopy(df00)

    # 참조 데이터프레임과 같이 다시 색인된 빈 데이터프레임을 생성합니다.
    df = pd.DataFrame().reindex_like(reference)
    pre_ind = df.index  # 원래 인덱스를 저장합니다.

    # 월별 다시 색인을 위해 인덱스를 수정합니다.
    df.index = df.index.month + df.index.year * 100
    df0.index = df0.index.month + df0.index.year * 100

    # 원본 데이터로 데이터프레임을 채웁니다.
    df = df.fillna(df0)

    # 정리하고 원래 인덱스를 복원합니다.
    del df0
    df.index = pre_ind

    return df


def resizer(df00: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    연간 또는 더 적게 업데이트된 데이터를 데이터프레임에 채웁니다.

    매개변수:
    df00 (pd.DataFrame): 원본 데이터가 있는 데이터프레임, 연간 또는 더 적게 업데이트됩니다.
    reference (pd.DataFrame): 연간 빈도로 다시 색인을 지정할 참조 데이터프레임.

    반환값:
    pd.DataFrame: 원본 데이터를 채운 후 연간 참조 데이터프레임에 맞게 정렬된 데이터프레임.
    """
    if not isinstance(df00.index, pd.DatetimeIndex) or not isinstance(reference.index, pd.DatetimeIndex):
        raise ValueError("입력하는 데이터프레임들의 인덱스는 Datetime index여야만 합니다.")

    # 원본 데이터프레임을 깊은 복사하여 수정하지 않도록 합니다.
    df0 = copy.deepcopy(df00)

    # 참조 데이터프레임과 같이 다시 색인된 빈 데이터프레임을 생성합니다.
    df = pd.DataFrame().reindex_like(reference)
    pre_ind = df.index  # 원래 인덱스를 저장합니다.

    # 연간 다시 색인을 위해 인덱스를 수정합니다.
    df.index = df.index.year
    df0.index = df0.index.year

    # 원본 데이터로 데이터프레임을 채웁니다.
    df = df.fillna(df0)

    # 정리하고 원래 인덱스를 복원합니다.
    del df0
    df.index = pre_ind

    return df


def exposure_calculation(index: pd.Series, returns: pd.DataFrame, period=60, min_period=36,
                         ff_model=None) -> pd.DataFrame:
    """
    특정 월별 인덱스 (e.g. KOSPI, S&P500)에 월별 주식 수익률이 노출된 정도를 측정합니다.

    매개변수:
    index (pd.Series): 월별 인덱스
    ret (pd.DataFrame): 행이 날짜고, 열이 종목인 월별 수익률 데이터프레임.
    period(int): 노출도를 계산하기 위한 관측치 수입니다. 기본적으로 60으로 설정되어 있습니다.
    min_period(int): 노출도를 계산하기 위해서 필요한 최소한의 관측치 수입니다. 기본적으로 36으로 설정되어 있습니다.
    ff_model (pd.DataFrame): 파마 프렌치 3요인, 5요인 및 모멘텀 팩터를 추가한 모델 등 리스크 조정용 리스크 팩터의 데이터프레임.

    반환값:
    pd.DataFrame: 월별 노출도 데이터프레임.
    """

    # 리스크 조정이 없는 경우: 단순히 입력 인덱스만 회귀분석에 쓴다.
    if ff_model is None:

        X = index.to_frame()
        X.columns = ["index"]

    else:

        X = ff_model.copy()
        X["index"] = index

    def regression(y):

        X0 = X.copy()

        X0["y"] = y

        X0.dropna(inplace=True)

        if X0.shape[0] < min_period:

            return np.nan

        else:

            # 리스크 조정 및 미조정 시에서의 데이터를 이용해 회귀분석하기
            return OLS(X0["y"], add_constant(X0.drop("y", axis=1))).fit().params["index"]

    return returns.rolling(period, min_periods=min_period).apply(regression)



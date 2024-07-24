
import statsmodels.api as smapi
import copy

import numpy as np
import pandas as pd

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



def factor_calculation(X: pd.DataFrame, nyse_filter: pd.DataFrame, market_cap_month_june: pd.DataFrame,
                       ret_month: pd.DataFrame, market_cap_month: pd.DataFrame, winsor_q=0.005) -> tuple:
    """
    주어진 기업 특성 데이터를 바탕으로 SMB와 특성 팩터를 계산하는 함수. (높은 것에서 낮은 걸 뺌)

    Args:
    X (pd.DataFrame): 분석 대상 데이터. 각 행은 날짜, 각 열은 개별 기업을 나타냅니다.
    nyse_filter (pd.DataFrame): 주요 시장 필터
    market_cap_month_june (pd.DataFrame): 6월의 시가총액 데이터. 각 행은 날짜, 각 열은 개별 기업의 시가총액을 나타냅니다.
    ret_month (pd.DataFrame): 월별 수익률 데이터. 각 행은 날짜, 각 열은 개별 기업의 수익률을 나타냅니다.
    market_cap_month (pd.DataFrame): 월별 시가총액 데이터. 각 행은 날짜, 각 열은 개별 기업의 시가총액을 나타냅니다.
    winsor_q (int): 수익률 윈저라이징 수준. raw number로 제시.

    Returns:
    tuple: SMB와 기업 특성 팩터를 포함하는 튜플 (smb, hml)
    """

    # 상위 30% 데이터 식별
    high = np.where(X > (X*nyse_filter).quantile(0.7, axis=1).values.reshape(-1, 1), 1, np.nan)
    # 하위 30% 데이터 식별
    low = np.where(X <= (X*nyse_filter).quantile(0.3, axis=1).values.reshape(-1, 1), 1, np.nan)
    # 중간 40% 데이터 식별
    middle = np.where((X > (X*nyse_filter).quantile(0.3, axis=1).values.reshape(-1, 1)) &
                      (X <= (X*nyse_filter).quantile(0.7, axis=1).values.reshape(-1, 1)), 1, np.nan)
    # 시가총액 상위 50% 데이터 식별
    big = np.where(market_cap_month_june > (market_cap_month_june*nyse_filter).quantile(0.5, axis=1).values.reshape(-1, 1), 1, np.nan)
    # 시가총액 하위 50% 데이터 식별
    small = np.where(market_cap_month_june <= (market_cap_month_june*nyse_filter).quantile(0.5, axis=1).values.reshape(-1, 1), 1, np.nan)

    # 각 그룹별 필터 생성
    sh = high * small
    sl = low * small
    sm = middle * small
    bh = high * big
    bl = low * big
    bm = middle * big

    result = []

    # 필터를 이용한 수익률 계산
    for filters in [sh, sl, sm, bh, bl, bm]:
        weighted_return = ((winsor(ret_month, winsor_q) * filters) * (market_cap_month * filters).shift(1))
        total_market_cap = (market_cap_month  * filters).sum(axis=1).shift(1).values.reshape(-1, 1)
        result.append((weighted_return / total_market_cap).sum(axis=1))

    # 결과 할당
    sh, sl, sm, bh, bl, bm = result

    # SMB (Small Minus Big) 계산
    smb = ((sh + sl + sm) / 3) - ((bh + bl + bm) / 3)
    # HML (High Minus Low) 계산
    hml = ((sh + bh) / 2) - ((sl + bl) / 2)

    return smb, hml



def fama_macbeth(dfxs: dict[str, pd.DataFrame], dfy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Fama-MacBeth 회귀 분석을 수행합니다.

    매개변수:
    dfxs (dict): 변수 이름을 나타내는 문자열을 키로 하고 독립 변수 데이터를 포함하는 pandas DataFrame을 값으로 가지는 딕셔너리.
                 각 DataFrame은 동일한 인덱스를 가져야 합니다.
    dfy (pd.DataFrame): dfxs의 DataFrame과 동일한 인덱스를 가진 종속 변수 데이터를 포함하는 pandas DataFrame.

    반환값:
    dict: 다음 키를 가진 딕셔너리:
        - "params": 각 기간에 대한 회귀 계수를 나타내는 DataFrame.
        - "num": 각 기간의 회귀에 사용된 관측치 수를 나타내는 Series.
        - "r2": 각 기간의 회귀에 대한 조정된 결정 계수 값을 포함하는 Series.
    """

    # 각 기간에 대한 회귀 계수 결과를 저장할 딕셔너리
    mid_result = {}

    # 각 기간의 관측치 수를 저장할 딕셔너리
    n_list = {}

    # 각 기간의 조정된 결정 계수 값을 저장할 딕셔너리
    rsqs = {}

    # 종속 변수 DataFrame의 각 기간 인덱스를 순회
    for ind in dfy.index:
        tmp = {}

        # 현재 기간에 대한 독립 변수 데이터를 수집
        for key in dfxs.keys():
            x = dfxs[key].loc[ind, :]
            tmp[key] = x

        # 현재 기간에 대한 종속 변수 데이터를 추가
        tmp["y"] = dfy.loc[ind, :]

        # 수집된 데이터로 DataFrame을 생성하고 결측값이 있는 행을 삭제
        tmp = pd.DataFrame(tmp).dropna()

        # 관측치 수가 변수 수보다 많은지 확인
        if tmp.shape[0] <= tmp.shape[1]:
            mid_result[ind] = np.nan
            continue

        # 수집된 데이터로 OLS 회귀 분석 수행
        model = smapi.OLS(tmp["y"], smapi.add_constant(tmp.drop("y", axis=1))).fit()

        # 회귀 계수, 관측치 수 및 조정된 결정 계수 값을 저장
        mid_result[ind] = model.params
        n_list[ind] = tmp.shape[0]
        rsqs[ind] = model.rsquared_adj

    # 결과 딕셔너리를 pandas DataFrame/Series로 변환
    mid_result = pd.DataFrame(mid_result).T

    return {
        "params": mid_result,
        "num": pd.Series(n_list, index=mid_result.index),
        "r2": pd.Series(rsqs, index=mid_result.index)
    }






def resizer_month(df00: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    월별로 데이터프레임을 재정렬하는 함수

    :param df00: 입력 데이터프레임
    :param reference: 참조 데이터프레임
    :return: 재정렬된 데이터프레임
    """
    df0 = copy.deepcopy(df00)
    df = pd.DataFrame().reindex_like(reference)
    pre_ind = df.index
    df.index = df.index.month + df.index.year * 100
    df0.index = df0.index.month + df0.index.year * 100
    df = df.fillna(df0)
    del df0
    df.index = pre_ind
    return df


def resizer(df00: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    년도별로 데이터프레임을 재정렬하는 함수

    :param df00: 입력 데이터프레임
    :param reference: 참조 데이터프레임
    :return: 재정렬된 데이터프레임
    """
    df0 = copy.deepcopy(df00)
    df = pd.DataFrame().reindex_like(reference)
    pre_ind = df.index
    df.index = df.index.year
    df0.index = df0.index.year
    df = df.fillna(df0)
    del df0
    df.index = pre_ind
    return df


def counter(val: pd.Series, quantiles: pd.DataFrame) -> pd.Series:
    """
    주어진 값이 퀀타일보다 작은 개수를 세는 함수

    :param val: 값 시리즈
    :param quantiles: 퀀타일 데이터프레임
    :return: 개수 시리즈
    """
    return (quantiles < val.values.reshape(-1, 1)).sum(axis=1) * np.where(val.isna(), np.nan, 1)


def quantile_maker(X: pd.DataFrame, K: pd.DataFrame, q: int) -> pd.DataFrame:
    """
    퀀타일을 생성하는 함수

    :param X: 입력 데이터프레임
    :param K: 기준 데이터프레임
    :param q: 퀀타일 수
    :return: 퀀타일 데이터프레임
    """
    qs_size = {}
    for n in range(1, q, 1):
        qs_size[n] = K.quantile(n / q, axis=1)
    qs_size = pd.DataFrame(qs_size)
    return X.apply(counter, quantiles=qs_size, axis=0) + 1


def y_maker_uni(
        vals: pd.DataFrame,
        observe_vals: dict[str, pd.DataFrame],
        nyse_filter: pd.DataFrame,
        ret: pd.DataFrame,
        market_cap: pd.DataFrame,
        q: int,
        label: str,
        equal_weight: bool = False,
        re_win: bool = True,
        threshold: float = 0.005
) -> dict[str, pd.DataFrame]:
    """
    단일 정렬을 수행하는 함수

    :param vals: 정렬할 값 데이터프레임
    :param observe_vals: 관찰할 값 데이터프레임 딕셔너리
    :param nyse_filter: NYSE 필터 데이터프레임
    :param ret: 수익률 데이터프레임
    :param market_cap: 시가총액 데이터프레임
    :param q: 퀀타일 수
    :param label: 레이블
    :param equal_weight: 동일 가중치 여부
    :param re_win: 윈저라이징 여부
    :param threshold: 윈저라이징 임계값
    :return: 결과 딕셔너리
    """
    result = {}
    X = vals.copy()
    q_filter = quantile_maker(X, X * nyse_filter, q=q)
    hundred_portfolios = {}
    nums = {}

    for key in observe_vals.keys():
        tmp_dict = {}
        for b_q in range(1, q + 1):
            filters = np.where(q_filter == b_q, 1, np.nan)
            tmp_dict[f"{label}/{b_q:01d}"] = ((observe_vals[key] * filters)).mean(axis=1)
            nums[f"{label}/{b_q:01d}"] = pd.DataFrame(filters, index=vals.index, columns=vals.columns).fillna(0).sum(axis=1)

            if equal_weight:
                if re_win:
                    hundred_portfolios[f"{label}/{b_q:01d}"] = winsor((ret * filters), threshold).mean(axis=1)
                else:
                    hundred_portfolios[f"{label}/{b_q:01d}"] = ((ret * filters)).mean(axis=1)
            else:
                if re_win:
                    hundred_portfolios[f"{label}/{b_q:01d}"] = (
                        (winsor(ret * filters, threshold) * (market_cap * filters).shift(1)) /
                        (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)
                else:
                    hundred_portfolios[f"{label}/{b_q:01d}"] = (
                        ((ret * filters) * (market_cap * filters).shift(1)) /
                        (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)

        result[key] = pd.DataFrame(tmp_dict)
    result["portRet"] = pd.DataFrame(hundred_portfolios)
    result["num"] = pd.DataFrame(nums)
    return result


def y_maker_bi_indi(
        vals1: pd.DataFrame,
        vals2: pd.DataFrame,
        observe_vals: dict[str, pd.DataFrame],
        nyse_filter: pd.DataFrame,
        ret: pd.DataFrame,
        market_cap: pd.DataFrame,
        q: int,
        label1: str,
        label2: str,
        equal_weight: bool = False,
        re_win: bool = True,
        threshold: float = 0.005
) -> dict[str, pd.DataFrame]:
    """
    독립적인 이중 정렬을 수행하는 함수

    :param vals1: 정렬할 값 데이터프레임 1
    :param vals2: 정렬할 값 데이터프레임 2
    :param observe_vals: 관찰할 값 데이터프레임 딕셔너리
    :param nyse_filter: NYSE 필터 데이터프레임
    :param ret: 수익률 데이터프레임
    :param market_cap: 시가총액 데이터프레임
    :param q: 퀀타일 수
    :param label1: 레이블 1
    :param label2: 레이블 2
    :param equal_weight: 동일 가중치 여부
    :param re_win: 윈저라이징 여부
    :param threshold: 윈저라이징 임계값
    :return: 결과 딕셔너리
    """
    result = {}
    X1 = vals1.copy()
    X2 = vals2.copy()
    q_filter1 = quantile_maker(X1, X1 * nyse_filter, q=q)
    q_filter2 = quantile_maker(X2, X2 * nyse_filter, q=q)
    hundred_portfolios = {}
    nums = {}

    for key in observe_vals.keys():
        tmp_dict = {}
        for s_q in range(1, q + 1):
            for b_q in range(1, q + 1):
                filters = np.where(q_filter1 == b_q, 1, np.nan) * np.where(q_filter2 == s_q, 1, np.nan)
                nums[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = pd.DataFrame(filters, index=vals1.index, columns=vals1.columns).fillna(0).sum(axis=1)
                tmp_dict[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = ((observe_vals[key] * filters)).mean(axis=1)

                if equal_weight:
                    if re_win:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = winsor((ret * filters), threshold).mean(axis=1)
                    else:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = ((ret * filters)).mean(axis=1)
                else:
                    if re_win:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = (
                            (winsor(ret * filters, threshold) * (market_cap * filters).shift(1)) /
                            (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)
                    else:
                        hundred_portfolios[f"{label1}{b_q:01d}/" f"{label2}{s_q:01d}"] = (
                            ((ret * filters) * (market_cap * filters).shift(1)) /
                            (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)

        result[key] = pd.DataFrame(tmp_dict)
    result["portRet"] = pd.DataFrame(hundred_portfolios)
    result["num"] = pd.DataFrame(nums)
    return result


def y_maker_bi_dep(
        controls: pd.DataFrame,
        vals: pd.DataFrame,
        observe_vals: dict[str, pd.DataFrame],
        nyse_filter: pd.DataFrame,
        ret: pd.DataFrame,
        market_cap: pd.DataFrame,
        q: int,
        label1: str,
        label2: str,
        equal_weight: bool = False,
        re_win: bool = True,
        threshold: float = 0.005
) -> dict[str, pd.DataFrame]:
    """
    종속적인 이중 정렬을 수행하는 함수

    :param controls: 통제 변수 데이터프레임
    :param vals: 정렬할 값 데이터프레임
    :param observe_vals: 관찰할 값 데이터프레임 딕셔너리
    :param nyse_filter: NYSE 필터 데이터프레임
    :param ret: 수익률 데이터프레임
    :param market_cap: 시가총액 데이터프레임
    :param q: 퀀타일 수
    :param label1: 레이블 1
    :param label2: 레이블 2
    :param equal_weight: 동일 가중치 여부
    :param re_win: 윈저라이징 여부
    :param threshold: 윈저라이징 임계값
    :return: 결과 딕셔너리
    """
    result = {}
    X1 = vals.copy()
    X2 = controls.copy()
    control_q_filter = quantile_maker(X2, X2 * nyse_filter, q=q)
    hundred_portfolios = {}
    nums = {}

    for key in observe_vals.keys():
        tmp_dict = {}
        for s_q in range(1, q + 1):
            filters_0 = np.where(control_q_filter == s_q, 1, np.nan)
            val_q_filter = quantile_maker(X1 * filters_0, X1 * filters_0, q=q)

            for b_q in range(1, q + 1):
                filters = np.where(val_q_filter == b_q, 1, np.nan)
                nums[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = pd.DataFrame(filters, index=controls.index, columns=controls.columns).fillna(0).sum(axis=1)
                tmp_dict[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = ((observe_vals[key] * filters)).mean(axis=1)

                if equal_weight:
                    if re_win:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = winsor((ret * filters), threshold).mean(axis=1)
                    else:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = ((ret * filters)).mean(axis=1)
                else:
                    if re_win:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = (
                            (winsor(ret * filters, threshold) * (market_cap * filters).shift(1)) /
                            (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)
                    else:
                        hundred_portfolios[f"{label1}/{s_q:01d}/" f"{label2}/{b_q:01d}"] = (
                            ((ret * filters) * (market_cap * filters).shift(1)) /
                            (market_cap * filters).sum(axis=1).shift(1).values.reshape(-1, 1)).sum(axis=1)

        result[key] = pd.DataFrame(tmp_dict)
    result["portRet"] = pd.DataFrame(hundred_portfolios)
    result["num"] = pd.DataFrame(nums)
    return result


class CrossPackage:
    returns: pd.DataFrame
    mkt_cap: pd.DataFrame
    mkt_name: pd.DataFrame
    characteristics: dict[str, pd.DataFrame]
    event: pd.DataFrame

    def __init__(
            self,
            returns: pd.DataFrame,
            mkt_cap: pd.DataFrame,
            mkt_name: pd.DataFrame,
            characteristics: dict[str, pd.DataFrame],
            main_market_name: str = "유가증권시장",
            event: pd.DataFrame = None,
            intersecting_data: bool = True
    ):
        """
        크로스 섹션 분석을 위한 패키지 클래스 초기화

        :param returns: 수익률 데이터프레임
        :param mkt_cap: 시가총액 데이터프레임
        :param mkt_name: 시장 이름 데이터프레임
        :param characteristics: 특성 데이터프레임 딕셔너리
        :param main_market_name: 주 시장 이름
        :param event: 이벤트 데이터프레임
        :param intersecting_data: 데이터 교차 여부
        """
        self.returns = returns
        self.mkt_cap = mkt_cap
        self.mkt_name = mkt_name
        self.characteristics = characteristics
        self.event = event

        if intersecting_data:
            inter_col = set(self.returns.columns)
            inter_ind = set(self.returns.index)
            inter_col = inter_col.intersection(set(self.mkt_cap.columns))
            inter_ind = inter_ind.intersection(set(self.mkt_cap.index))

            for key in self.characteristics.keys():
                inter_col = inter_col.intersection(set(self.characteristics[key].columns))
                inter_ind = inter_ind.intersection(set(self.characteristics[key].index))

            inter_col = sorted(list(inter_col))
            inter_ind = sorted(list(inter_ind))

            self.returns = self.returns.loc[inter_ind, inter_col]
            self.mkt_cap = self.mkt_cap.loc[inter_ind, inter_col]
            self.mkt_name = self.mkt_name.loc[inter_ind, inter_col]
            self.main_mrt = pd.DataFrame(np.where(self.mkt_name == main_market_name, 1, np.nan), index=self.mkt_name.index, columns=self.mkt_name.columns).loc[inter_ind, inter_col]

            for key in self.characteristics.keys():
                self.characteristics[key] = self.characteristics[key].loc[inter_ind, inter_col]

        else:
            print("Warning. If the data does not intersect, further analysis cannot be conducted. Consider intersecting_data=True")

    def single_sort(
            self,
            sort_val: str,
            observe_vals: list[str],
            re_win: bool = True,
            q: int = 10,
            ew: bool = True,
            threshold: float = 0.005
    ) -> dict[str, pd.DataFrame]:
        """
        단일 정렬을 수행하는 메소드

        :param sort_val: 소팅 변수 이름
        :param observe_vals: 관찰 변수 리스트
        :param re_win: 수익률 윈저라이징 여부 (불린)
        :param q: 분할 개수
        :param ew: 동일가중 여부
        :param threshold: 윈저라이징 임계값
        :return: 분석 결과 (관찰 변수 + portRet)
        """
        return y_maker_uni(self.characteristics[sort_val], {val: self.characteristics[val] for val in observe_vals}, self.main_mrt, self.returns, self.mkt_cap, q, sort_val, equal_weight=ew, re_win=re_win)

    def double_independent_sort(
            self,
            sort1_val: str,
            sort2_val: str,
            observe_vals: list[str],
            q: int = 10,
            re_win: bool = True,
            ew: bool = True,
            threshold: float = 0.005
    ) -> dict[str, pd.DataFrame]:
        """
        독립적인 이중 정렬을 수행하는 메소드

        :param sort1_val: 독립 소팅 변수 1 이름
        :param sort2_val: 독립 소팅 변수 2 이름
        :param observe_vals: 관찰 변수 리스트
        :param q: 분할 개수
        :param re_win: 수익률 윈저라이징 여부 (불린)
        :param ew: 동일가중 여부
        :param threshold: 윈저라이징 임계값
        :return: 분석 결과 (관찰 변수 + portRet)
        """
        return y_maker_bi_indi(self.characteristics[sort1_val], self.characteristics[sort2_val], {val: self.characteristics[val] for val in observe_vals}, self.main_mrt, self.returns, self.mkt_cap, q, sort1_val, sort2_val, equal_weight=ew, re_win=re_win)

    def double_dependent_sort(
            self,
            control_val: str,
            sort_val: str,
            observe_vals: list[str],
            q: int = 10,
            re_win: bool = True,
            ew: bool = True,
            threshold: float = 0.005
    ) -> dict[str, pd.DataFrame]:
        """
        종속적인 이중 정렬을 수행하는 메소드

        :param control_val: 통제 변수
        :param sort_val: 소팅 변수
        :param observe_vals: 관찰 변수 리스트
        :param q: 분할 개수
        :param re_win: 수익률 윈저라이징 여부 (불린)
        :param ew: 동일가중 여부
        :param threshold: 윈저라이징 임계값
        :return: 분석 결과 (관찰 변수 + portRet)
        """
        return y_maker_bi_dep(self.characteristics[control_val], self.characteristics[sort_val], {val: self.characteristics[val] for val in observe_vals}, self.main_mrt, self.returns, self.mkt_cap, q, control_val, sort_val, equal_weight=ew, re_win=re_win)

    def fama_macbeth(self, variables: list[str], threshold: float) -> dict[str, pd.DataFrame]:
        """
        Fama-MacBeth 회귀분석을 수행하는 메소드

        :param variables: 변수 리스트
        :param threshold: 윈저라이징 임계값
        :return: 회귀분석 결과 딕셔너리
        """
        tmp_xs = {}
        for var in variables:
            tmp_xs[var] = winsor(self.characteristics[var], threshold=threshold)
        return fama_macbeth(tmp_xs, winsor(self.returns, threshold=threshold) * 100)


    def factor_generation(self, variable, q=0.005):

        return factor_calculation(self.characteristics[variable], self.main_mrt, resizer(self.mkt_cap.shift(6), self.returns).shift(6), self.returns, self.mkt_cap, q)[1]


def portfolios_to_tables(data0: pd.Series, col_label: str, row_label: str, divider: str = "/",
                         index_format: str = "lnln") -> pd.DataFrame:
    """
    포트폴리오 통계량에서 나온 데이터 시리즈를 테이블 형태로 변환하는 함수

    :param data0: 입력 데이터 시리즈 (각 이중 정렬 포트폴리오에서의 통계량)
    :param col_label: 컬럼 레이블 (컬럼으로 사용할 레이블)
    :param row_label: 행 레이블 (행으로 사용할 레이블)
    :param divider: 인덱스 구분자 (기본값: "/")
    :param index_format: 인덱스 포맷, 'l'은 레이블, 'n'은 숫자 (기본값: "lnln")
    :return: 피벗 테이블 형태의 데이터프레임
    """

    # 입력 데이터 복사 및 데이터프레임으로 변환
    data = data0.copy().to_frame()

    # 인덱스 포맷에 따른 위치 변수 초기화
    label1_index = None
    label2_index = None
    num1_index = None
    num2_index = None

    # 인덱스 포맷의 길이가 4인지 확인
    assert len(index_format) == 4, "index_format 길이는 반드시 4이어야 합니다."

    # 인덱스 포맷을 기반으로 위치 변수 설정
    for n in range(4):
        if index_format[n] == "l" and label1_index is None:
            label1_index = n
        elif index_format[n] == "l" and label1_index is not None:
            label2_index = n
        elif index_format[n] == "n" and num1_index is None:
            num1_index = n
        elif index_format[n] == "n" and num1_index is not None:
            num2_index = n

    # 인덱스를 구분자로 분할하여 새로운 컬럼 생성
    new_cols = pd.DataFrame(pd.Series(data.index).str.split(divider, expand=True).values, index=data.index)

    # 새로운 컬럼의 인덱스를 기반으로 입력 배열 설정
    if col_label not in list(new_cols.iloc[:, label1_index]):
        inputs = [label2_index, num1_index, num2_index]
    else:
        inputs = [label1_index, num2_index, num1_index]

    # 데이터에 새로운 컬럼 추가
    data[["col_name", row_label, col_label]] = new_cols.iloc[:, inputs]

    # 피벗 테이블 생성 및 반환
    return data[data["col_name"] == col_label].pivot(columns=col_label, index=row_label, values=data.columns[0])

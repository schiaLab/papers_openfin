import matplotlib.pyplot as plt

import os


def set_korean_plt():

    if os.name == 'posix':  # Mac 환경 폰트 설정
        plt.rc('font', family='AppleGothic')
    elif os.name == 'nt':  # Windows 환경 폰트 설정
        plt.rc('font', family='Malgun Gothic')

    plt.rc('axes', unicode_minus=False)

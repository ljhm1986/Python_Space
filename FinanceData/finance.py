# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:24:52 2019

@author: stu12
"""

import pandas as pd

KTPData = pd.read_excel("C:\\WorkSpace\\FinanceBigData\\KT_유동인구.xlsx",
                      sheet_name = '유동인구 데이터')
type(KTPData)
KTPData.info()
KTPData.astype
KTPData.describe()

#null 이 있나? 보니까 없다. 
KTPData.isnull().sum()

KTPData.columns

#여러 지역
KTPData['AMD_CD'].unique()

#2019년 6월1일 ~ 6월 30일 
KTPData['etl_ymd'].unique()

#시간 0시 ~ 23시
KTPData['timezn_cd'].unique()

#요일별 평균 유동인구
#시간별 평균 유동인구 구해볼까?

a = KTPData['total'].groupby(KTPData['etl_ymd']).sum()
a
pd.set_option('display.max_rows',100)
type(a)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties \
(fname ="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family = font_name)

plt.plot(a)

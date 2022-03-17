# import pandas as pd
# import matplotlib.pyplot as plt
# from xlrd import xldate_as_tuple
# from datetime import datetime
# 时序图
# data = pd.read_excel('./11-1.1.xlsx', index_col="日期")
# # data.index = pd.to_datetime(data.index)
# 先转换为dadetime形式，再利用strftime转为指定格式
# data.index = [datetime(*xldate_as_tuple(i, 0)).strftime('%Y-%m') for i in data.index.values]
# data1, data2 = data.loc[:, ["出口总额"]], data.loc[:, ["外汇储备"]]
# data1.plot(figsize=(18, 12))
# data2.plot(figsize=(18, 12))
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()

# # 线性拟合
# import math
# import pandas as pd
# import numpy as np
# from statsmodels.formula.api import ols
# import matplotlib.pyplot as plt
# data = pd.read_excel('./11-1.2.xlsx')
# data['季度'] = np.arange(1, 41)
# model = ols('消费支出数据~季度', data).fit()
# y = data['消费支出数据'].values
# y_prd = model.predict(data['季度'])
# e = y - y_prd
# sigama = math.sqrt(np.var(e)*40/38)
# print(sigama)
# print(model.summary())
# # 曲线拟合
# data1 = data.loc[:, ["消费支出数据"]]
# data1.plot(figsize=(18, 12))
# plt.xticks(np.arange(0, 50, 10))
# plt.xlabel('Time')
# plt.ylabel('消费支出数据(单位：百万澳元)')
# x2 = [0, 40]
# y2 = [model.params[0] + model.params[1]*i for i in x2]
# plt.plot(x2, y2, 'r-')
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
# 多项式曲线拟合
# import math
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import leastsq
# data = pd.read_excel('./11-1.3.xlsx')
# x, y = np.arange(1, 61), np.array(data['化肥产量'].values)
# coef = np.polyfit(x, y, 2)
# p1 = np.poly1d(coef)
# print(p1)
# y_fit = np.polyval(coef, x)
# e = y - y_fit
# sigama = math.sqrt(np.var(e)*60/57)
# print(sigama)
# plt.figure
# t = np.array(data['年份'].values)
# plt.scatter(t, y, label='Original scatter figure')
# plt.plot(t, y_fit, '-b', label='Fitted curve')
# plt.legend()
# plt.show()

# # 移动平均法预测
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# data = pd.read_excel('./11-1.4.xlsx')
# y = np.array(data['温度'].values)
# def moving_average(a, n):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# latter = moving_average(y, n=5)
# t1 = np.array(data['年份'].values)
# t2 = np.array(data['年份'].values)[4:]
# plt.plot(t1, y, label='Original curve')
# plt.plot(t2, latter, '-r', label='Moving Average curve')
# plt.legend()
# plt.show()

# holt两参数指数平滑法
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

# data = pd.read_excel('./11-1.5.xlsx')
# data_sr = data['沙产量'].values
# t1 = np.array(data['年份'].values)
# t2 = np.hstack((t1, np.array([2000, 2001, 2002, 2003, 2004,
#                              2005, 2006, 2007, 2008, 2009])))
# # Holt’s Method
# fit1 = Holt(data_sr).fit(smoothing_level=0.855644, smoothing_trend=0.158537, optimized=False)
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(t1, data_sr)
# ax[0].plot(t1, list(fit1.fittedvalues))
# ax[1].plot(t2, list(fit1.fittedvalues) + list(fit1.forecast(10)), '-r', marker='^')
# print(fit1.forecast(10))
# plt.show()
# Holt-Winters三参数指数平滑
# data = pd.read_excel('./11-1.6.xlsx')
# data_sr1 = data['产奶量'].values
# fit2 = ExponentialSmoothing(data_sr1, seasonal_periods=12, trend='add', seasonal='add').fit()
# t4 = np.append(np.array(data['年'].unique()).reshape(7, 2)[:, 0], [1976])
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(data_sr1)
# ax[0].set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
# ax[0].set_xticklabels(t4)
# ax[0].plot(list(fit2.fittedvalues))
# ax[1].plot(list(fit2.fittedvalues) + list(fit2.forecast(10)), '-r')
# ax[1].set_xticks([36, 97, 156])
# ax[1].set_xticklabels([1965, 1970, 1975])
# print(fit2.forecast(10))
# plt.show()

# 1.7时序图
# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./11-1.7.xlsx', index_col='月份')
# data_melt = data.melt()
# x = np.append(np.array(data_melt['variable'].unique()), [2001])
# y = data_melt['value']
# plt.plot(y, marker='^')
# plt.xticks([0, 12, 24, 36, 48, 60, 71], x)
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('Temperature')
# plt.show()

# import matplotlib.pyplot as plt
# x = range(1, 13)
# y = [-3.1,1.02,7.18,14.58,20.2,25,27.3,25.38,20.53,13.62,5.03,-0.35]
# plt.plot(x, y, marker='o')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show()

# 时序图+时间序列分解
# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./11-1.8.xlsx', index_col='月份')
# data_melt = data.melt()
# data2 = pd.concat([data_melt['variable'], data_melt['variable'].str.split('年', expand=True)], axis=1)
# data00 = pd.Series(['01','02','03','04','05','06','07','08',
#                     '09','10','11','12'])
# data00e = pd.to_datetime(data00).apply(lambda x: x.strftime('%m'))
# data_melt.index = pd.to_datetime(data_melt[0]).apply(lambda x: x.strftime('%Y'))
# index = data2[0]+data00
# del data_melt['variable']
# x = np.append(np.array(data_melt['variable'].unique()), ['2001年'])
# y = data_melt['value']
# plt.plot(y)
# plt.xticks([0, 12, 24, 36, 48, 60, 72, 84, 95], x)
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(data_melt['value'], model="multiplicative", period=12)
# # result.observed.plot()
# # result.trend.plot()
# result.resid.plot()
# # result.seasonal.plot() # 综合图
# plt.xticks([0, 20, 40, 60, 80], labels=[1993, 1995, 1997, 1999, 2001])
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()

# AR模型ACF和PACF图
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# num = 100
# np.random.seed(10)
# e = np.random.randn(num)
# x0, x1, x2, x3 = np.empty(num), np.empty(num), np.empty(num), np.empty(num)
# x0[0], x1[0], x2[0], x3[0] = 2, 2, 2, 2
# # 生成AR（1）序列
# for i in range(1, num):
#     x0[i] = 0.8 * x0[i - 1] + e[i]
#     x1[i] = -1.1 * x1[i - 1] + e[i]
# plt.subplot(221, title="AR({0}):x[t]={1}*x[t-1]+e".format(1,0.8))
# plt.plot(x0)
# plt.subplot(222, title="AR({0}):x[t]={1}*x[t-1]+e".format(1,-1.1))
# plt.plot(x1)
# # plot_pacf(x0)
# # plot_pacf(x1)
# # 生成AR（2）序列
# for i in range(2, num):
#     x2[i] = x2[i - 1] - 0.5 * x2[i - 2] + e[i]
#     x3[i] = x3[i - 1] + 0.5 * x3[i - 2] + e[i]
# plt.subplot(223, title="AR({0}):x[t]={1}*x[t-1]-{2}*x[t-2]+e".format(2, 1, 0.5))
# plt.plot(x2)
# plt.subplot(224, title="AR({0}):x[t]={1}*x[t-1]+{2}*x[t-2]+e".format(2, 1, 0.5))
# plt.plot(x3)
# # plot_pacf(x2)
# # plot_pacf(x3)
# plt.tight_layout()
# plt.show()

# # MA模型
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# num = 1000
# e = np.empty(num+1)
# x0, x1, x2, x3 = np.empty(num), np.empty(num), np.empty(num), np.empty(num)
# for i in range(num+1):
#     e[i] = np.random.randn(num+1)[0]
# for i in range(num):
#     x0[i] = e[i] - 2 * e[i-1]
#     x1[i] = e[i] - 0.5 * e[i - 1]
# # plot_pacf(x0)
# # plot_pacf(x1)
# for i in range(num-2):
#     x2[i] = e[i] - 4/5 * e[i-1] + 16/25 * e[i-2]
#     x3[i] = e[i] - 5/4 * e[i - 1] + 25/16 * e[i-2]
# # plot_pacf(x2)
# plot_pacf(x3)
# plt.show()

# # ARMA模型
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# num = 1000
# np.random.seed(10)
# x0, e = np.empty(num), np.empty(num)
# x0[0] = 2
# for i in range(num):
#     e[i] = np.random.randn(num)[0]
# for i in range(1, num):
#     x0[i] = 0.5 * x0[i - 1] + e[i] + 0.8
# plot_acf(x0)
# plot_pacf(x0)
# plt.show()

# 1.18时序图
import pandas as pd
import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.1.xlsx', index_col='年份')
# data.plot(marker='^')
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# # 白噪声检验
# from statsmodels.stats.diagnostic import acorr_ljungbox
# y = data['新增里程'].values
# result = acorr_ljungbox(y, lags=[6, 12])
# print(result)
# '''data:  x
# X-squared = 37.754, df = 6, p-value = 1.255e-06
# data:  x
# X-squared = 44.62, df = 12, p-value = 1.197e-05'''
# # (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# plot_acf(y)
# plot_pacf(y)
# plt.show()
# # 自动定阶
# import statsmodels.api as sm
# import pmdarima as pm
# model = pm.auto_arima(y, seasonal=True, m=12)
# model.fit(y)
# print(model.summary())
# # 为了控制计算量，我们限制AR最大阶不超过6，MA最大阶不超过4。
# ar = sm.tsa.arma_order_select_ic(y,max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC
# print(ar)
# 模型拟合+参数估计
# from statsmodels.tsa.arima_model import ARMA
# model = ARMA(data, order=(2, 0)).fit()
# print(model.summary())        # 生成一份模型报告
# 模型检验（残差序列检验）
# import statsmodels.api as sm
# res = model.resid
# r, q, p = sm.tsa.acf(res.values.squeeze(), qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))

# 模型参数检验
# from scipy.stats import t
# t1, t2, t3 = 0.7185/0.108, 0.5294/0.107, 11.0227/3.091
# t1, t2, t3 = t.cdf(-t1, 56), t.cdf(-t2, 56), t.cdf(-t3, 56)
# print(t1, t2, t3)

# 例1.19
# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.2.xlsx', header=None)
# data0 = pd.DataFrame(data.values.T)
# data = data0.melt()
# del data['variable']
# data.plot(marker='^')
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# # 白噪声检验
# from statsmodels.stats.diagnostic import acorr_ljungbox
# y = data['value'].values
# result = acorr_ljungbox(y, lags=[6, 12], boxpierce=True)
# print(result)
# '''data:  x
# X-squared = 20.209, df = 6, p-value = 0.002542
# data:  x
# X-squared = 21.622, df = 12, p-value = 0.04198'''
# (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# plot_acf(y)
# plot_pacf(y)
# plt.show()
# 模型拟合+参数估计
from statsmodels.tsa.arima_model import ARMA
# model = ARMA(data, order=(0, 2)).fit()
# print(model.summary())        # 生成一份模型报告
# 模型检验（残差序列检验）
import statsmodels.api as sm
# res = model.resid
# r, q, p = sm.tsa.acf(res.values.squeeze(), qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))
# model2 = ARMA(data, order=(1, 0)).fit()  #拟合ＡR(1)模型
# print(model2.summary())        # 生成一份模型报告
# res2 = model2.resid
# r, q, p = sm.tsa.acf(res2.values.squeeze(), qstat=True)
# data2 = np.c_[range(1, 41), r[1:], q, p]
# table2 = pd.DataFrame(data2, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table2.set_index('lag'))  #AR(1)模型显著性检验

# # 例1.8预测
# result = model.forecast(5)
# print(result)  #为未来5天进行预测， 返回预测结果， 标准误差， 和置信区间
# data0 = pd.DataFrame(result[0], index=[2009, 2010, 2011, 2012, 2013], columns=['predict'])
# data_pred = pd.concat([data, data0], axis=1)
# data_pred.plot()
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()

# 例1.20
import pandas as pd
import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.3.xlsx', index_col='年份')
# data.plot(marker='^')
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# 一阶差分
# y = data["农业"].diff(1).dropna()
# plt.figure(figsize=(10, 6))
# y.plot()
# plt.xlabel('年份',fontsize=12, verticalalignment='top')
# plt.ylabel('农业差分',fontsize=14, horizontalalignment='center')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()
# (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# y1 = y.values
# plot_acf(y1)
# plot_pacf(y1)
# plt.show()
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(data, order=(0, 1, 1)).fit()
# print(model.summary())
# 模型检验（残差序列检验）
# import statsmodels.api as sm
# res = model.resid
# r, q, p = sm.tsa.acf(res.values.squeeze(), qstat=True)
# data = np.c_[range(1, 36), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))
# 模型预测
# result = model.forecast(10)
# print(result[0])
# data0 = pd.DataFrame(result[0], index=range(1989, 1999), columns=['predict'])
# data_pred = pd.concat([data, data0], axis=1)
# data_pred.plot()
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()

# 例1.21
# 疏系数ARIMA模型
import pandas as pd
# import matplotlib.pyplot as plt
data = pd.read_excel('./12-1.4.xlsx', index_col='年份')
# data.plot(marker='^')
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# 一阶差分
# y = data["每万人生育率"].diff(1).dropna()
# plt.figure(figsize=(10, 6))
# y.plot()
# plt.xlabel('Time',fontsize=12, verticalalignment='top')
# plt.ylabel('xdiff',fontsize=14, horizontalalignment='center')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()
# (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# y1 = y.values
# plot_acf(y1)
# plot_pacf(y1)
# plt.show()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data, order=(4, 1, 0), exog=(None,0,0,None)).fit()
print(model.summary())

# 例1.16
# SARIMA模型
# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.5.xlsx', header=None)
# data0 = pd.DataFrame(data.values.T)
# data = data0.melt()
# index0 = []
# for i in range(1942, 1972):
#     index0 = index0 + [i]*4
# del data['variable']
# data.plot()
# plt.xticks(range(12, 113, 20), labels=range(1945, 1971, 5))
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# 一阶4步差分
# y = data["value"].diff(1).diff(4).dropna()
# plt.figure(figsize=(10, 6))
# y.plot()
# plt.xticks(range(12, 113, 20), labels=range(1945, 1971, 5))
# plt.xlabel('Time',fontsize=12, verticalalignment='top')
# plt.ylabel('xdiff',fontsize=14, horizontalalignment='center')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()
# (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# y1 = y.values
# plot_acf(y1)
# plot_pacf(y1)
# plt.show()
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(y, order=(1, 0, 1)).fit()
# print(model.summary())
# # 模型选择（调参）
# import statsmodels.api as sm
# import warnings
# import itertools
# y = data["value"].values
# p = d = q = range(0, 2)
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], 0, x[2], 4) for x in list(itertools.product(p, d, q))]
# warnings.filterwarnings("ignore")  # specify to ignore warning messages
# aic = []
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit(disp=False)
#             aic.append(results.aic)  # 通过两次训练可以得到aic最小值，并打印出来合适的模型
#             if results.aic == 59.77432996642648:
#                 print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue
# # 得到最小AIC的模型以及对应AIC的值：ARIMA(1, 1, 0)x(1, 0, 1, 4) - AIC:59.77432996642648

# 例1.17
# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.6.xlsx', index_col='年份')
# data0 = pd.DataFrame(data.values.T)
# data = data0.melt()
# del data['variable']
# data.plot()
# plt.xticks(range(24, 395, 60), labels=range(1950, 1985, 5))
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.show()
# 一阶12步差分
# y = data["value"].diff(1).diff(12).dropna()
# plt.figure(figsize=(10, 6))
# y.plot()
# plt.xticks(range(24, 395, 60), labels=range(1950, 1985, 5))
# plt.xlabel('Time',fontsize=12, verticalalignment='top')
# plt.ylabel('xdiff',fontsize=14, horizontalalignment='center')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.show()
# # (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# y1 = y.values
# plot_acf(y1)
# plot_pacf(y1)
# plt.show()
# 模型拟合+残差检验
# import statsmodels.api as sm
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(y, order=(1, 0, 1)).fit()
# res = model.resid
# r, q, p = sm.tsa.acf(res.values.squeeze(), qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))
# import statsmodels.api as sm
# train = data[:200]
# test = data[200:]
# model2 = sm.tsa.statespace.SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)
# results = model2.fit(disp=False)
# print(results.summary())
# res1 = results.predict(start=200, end=407, dynamic=True)
# x, x_diff = test['value'].values, res1.values
# C = []
# for i in range(len(x)-1):
#     c = x[i]+x_diff[i+1]
#     C.append(c)
# C = [0]*1+C
# resid = test['value'].values - C
# #白噪声检验
# r, q, p = sm.tsa.acf(resid.squeeze(), qstat=True)
# data = np.c_[range(1, 41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))

# 残差自回归模型
# 线性拟合
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# data = pd.read_excel('./12-1.3.xlsx')
# X = data['Year'].values
# Y = data['Agriculture'].values
# split dataset
# x1 = X.reshape((len(X), 1))
# y1 = Y.reshape((len(X), 1))
# 转换成numpy的ndarray数据格式，n行1列,LinearRegression需要列格式数据，如下：
# X_test = X.reshape((len(X), 1))[len(X)-15:]
# Y_test = Y.reshape((len(X), 1))[len(X)-15:]
# 新建一个线性回归模型，并把数据放进去对模型进行训练
# lineModel = LinearRegression()
# lineModel.fit(x1, y1)
# Y_predict = lineModel.predict(X_test)
# plt.scatter(X_test, Y_test, c="blue")
# plt.plot(X_test, Y_predict, c="yellow")
# 残差序列DW检验
# et = Y_test - Y_predict
# et = et.flatten().tolist()
#D-W统计量
# t = len(et)
# et_1 = et[1:t+1]
# et_change = et[0:t]
# DW = sum([(m-n)**2 for m,n in zip(et_change,et_1)])/sum([m**2 for m in et_change])
# print(DW)

# 曲线为关于延迟变量的自回归拟合
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
# train autoregression
# train_X, test_X = X[:len(X)-15], X[len(X)-15:]
# train, test = Y[:len(X)-15], Y[len(X)-15:]
# model = AR(train)
# model_fit = model.fit()
# print(model_fit.summary())
# et = model_fit.resid.tolist()
# window = model_fit.k_ar
# coef = model_fit.params
# # walk forward over time steps in test
# history = train[len(train)-window:]
# history = [history[i] for i in range(len(history))]
# predictions = list()
# for t in range(len(test)):
#     length = len(history)
#     lag = [history[i] for i in range(length-window,length)]
#     yhat = coef[0]
#     for d in range(window):
#         yhat += coef[d+1] * lag[window-d-1]
#     obs = test[t]
#     predictions.append(yhat)
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# # plot
# pyplot.plot(test_X, test)
# pyplot.plot(test_X, predictions, color='red')
# pyplot.show()

# DW DH统计量
# import numpy as np
# # 模型延迟因变量系数的最小乘估计的方差
# var0 = sum(np.array([0.282,0.654,0.861,0.945,0.911,0.857,0.689,0.392])**2)/5
# t = len(et)
# et_1 = et[1:t+1]
# et_change = et[0:t]
# DW = sum([(m-n)**2 for m,n in zip(et_change,et_1)])/sum([m**2 for m in et_change])
# Dh = DW*t/(1-t*var0)
# print(DW)
# print(Dh)

# 转换成numpy的ndarray数据格式，n行1列,LinearRegression需要列格式数据，如下：
# X_train = X.reshape((len(X), 1))
# Y_train = Y.reshape((len(Y), 1))
# # 新建一个线性回归模型，并把数据放进去对模型进行训练
# lineModel = LinearRegression()
# lineModel.fit(X_train, Y_train)
# Y_predict = lineModel.predict(X_train)
# et = Y_predict - Y_train
# et = et.flatten().tolist()
# data1 = pd.DataFrame({'year': X, 'value': et})  # 残差序列
# (偏)自相关系数
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# y1 = data1['value'].values
# plot_acf(y1)
# plot_pacf(y1)
# plt.show()
# 模型拟合+残差检验
# import statsmodels.api as sm
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(y1, order=(2, 0, 0)).fit()
# print(model.summary())
# res = model.resid
# r, q, p = sm.tsa.acf(res.squeeze(), qstat=True)
# data = np.c_[range(1, 37), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))
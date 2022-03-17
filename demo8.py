import math

import numpy as np
from sympy import symbols, Eq, solve
# 矩估计
# data = np.array([3.23, 3.90, 4.75, 13.41, 15.94, 18.65, 18.99, 5.66, 2.80, 9.05])
# a, b = symbols('a, b')
# eqs = [Eq((a+b)/2, data.mean()), Eq((a**2+a*b+b**2)/3, (data**2).mean())]
# result = solve(eqs, [a, b])
# for value in result:
#     if value[0] <= value[1]:
#         print('a:{}'.format(value[0]))
#         print('b:{}'.format(value[1]))
# # 极大似然估计
# import sympy, scipy.stats
# from sympy import *
# from scipy.optimize import fsolve
# data = np.array([0.16, 0.56, 1.59, 0.84, -1.73, 0.65, 2.96, 1.04, 2.41, 0.94, 192.40, -2.89])
# n = len(data)
# theta = symbols('theta')
# y = n*math.log(math.pi) - sum([sympy.log(1+(i-theta)**2) for i in data])
# sol, = sympy.solve(sympy.diff(y, theta), theta)
# print(sol)
# f = sympy.diff(y, theta)
# print(fsolve(f, [0]))

# # 置信区间
# import scipy.stats as st
# import numpy as np
# import math
# data = np.array([23,35,39,27,36,44,36,42,46,43,31,33,
#         42,53,45,54,47,24,34,28,39,36,44,40,
#         39,49,38,34,48,50,34,39,45,48,45,32])
# z = st.norm.ppf(0.95, 0, 1)
# mean = data.mean()
# n = len(data)
# std = math.sqrt(sum([(i-mean)**2 for i in data])/(n-1))
# print(mean + z*std/math.sqrt(n), mean - z*std/math.sqrt(n))

# # t分布置信区间
# import scipy.stats as st
# import numpy as np
# import math
# data = np.array([1510,1450,1480,1460,1520,1480,1490,1460,
#         1480,1510,1530,1470,1500,1520,1510,1470])
# n = len(data)
# t = st.t.ppf(0.025, n-1)
# mean = data.mean()
# std = math.sqrt(sum([(i-mean)**2 for i in data])/(n-1))
# print(mean + t*std/math.sqrt(n), mean - t*std/math.sqrt(n))

# # 甲、乙两个机场平均得分之差的95%的置信区间
# import scipy.stats as st
# import numpy as np
# # 整理数据
# data = []
# with open('./甲乙公司.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         arr = line[:-1].split('\n\t')
#         for brr in arr:
#             sub_brr = brr[:-1].split('	')
#         if sub_brr:
#             data.append(sub_brr)
# num = []
# for num_0 in data:
#     num_0 = list(map(float, num_0))
#     num.append(num_0)
# data1 = np.array(num[0])
# data2 = np.array(num[1])
# n1, n2 = len(data1), len(data2)
# mean1 = data1.mean()
# mean2 = data2.mean()
# mean = mean1 - mean2
# var1 = data1.var()*(n1/(n1-1))
# var2 = data2.var()*(n2/(n2-1))
# z = st.norm.ppf(0.975, 0, 1)
# print(mean - z*math.sqrt(var1/n1 + var2/n2), mean + z*math.sqrt(var1/n1 + var2/n2))

# # 求解均值之差置信区间
# from scipy import stats
# import numpy as np
# def confidence_interval_udif(data1, data2, sigma1, sigma2, alpha=0.05):
#     xb1 = np.mean(data1)
#     xb2 = np.mean(data2)
#     n1 = len(data1)
#     n2 = len(data2)
#
#     if sigma1 > 0 and sigma2 > 0:  # 方差已知
#         tmp = np.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)
#         Z = stats.norm(loc=0., scale=1.)
#         return ((xb1 - xb2) + tmp * Z.ppf(alpha / 2), (xb1 - xb2) - tmp * Z.ppf(alpha / 2))
#     else:  # 方差未知
#         if sigma1 == sigma2:  # 未知且相等
#             sw = ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2)
#             tmp = np.sqrt(sw) * np.sqrt(1 / n1 + 1 / n2)
#             T = stats.t(df=n1 + n2 - 2)
#             return ((xb1 - xb2) + tmp * T.ppf(alpha / 2), (xb1 - xb2) - tmp * T.ppf(alpha / 2))
#         else:  # 未知且不等
#             tmp = np.sqrt(np.var(data1, ddof=1) / n1 + np.var(data2, ddof=1) / n2)
#             T = stats.t(df=13)
#             return ((xb1 - xb2) + tmp * T.ppf(alpha / 2), (xb1 - xb2) - tmp * T.ppf(alpha / 2))
# # 整理数据
# data = []
# with open('./方法一二.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         arr = line[:-1].split('\n\t')
#         for brr in arr:
#             sub_brr = brr[:-1].split('	')
#         if sub_brr:
#             data.append(sub_brr)
# num = []
# for num_0 in data:
#     num_0 = list(map(float, num_0))
#     num.append(num_0)
# num = np.array(num)
# data1, data2 = num[:, 0], num[:, 1]
# # 方差相等
# print(confidence_interval_udif(data1, data2, -1, -1, 0.05))
# # 方差不相等
# print(confidence_interval_udif(data1, data2, -1, -2, 0.05))

# # 配对样本均值之差检验
# import numpy as np
# import pandas as pd
# import math
# import scipy.stats as st
# data = pd.read_excel('./工艺.xlsx')
# ser = data['传统工艺']-data['新工艺']
# test = np.array(ser)
# df = len(test)
# mean = np.mean(test)
# std = np.std(test, ddof=1)
# ci = st.t.interval(alpha=0.95, df=df-1, loc=mean, scale=std/math.sqrt(df))
# print(ci)  # 双侧置信区间
# t = st.t.isf(0.05, df-1)
# print(mean - t*std/math.sqrt(df))  # 单侧下限置信区间

# # 比例置信区间估计
# import math
# import scipy.stats as st
# n = 500
# p = 325/500
# std = math.sqrt(p*(1-p)/n)
# ci = st.norm.interval(alpha=0.95, loc=p, scale=std)
# print(ci)
# # 现代方法估计
# n = 500 + 4
# p = (325+2)/n
# std = math.sqrt(p*(1-p)/n)
# ci = st.norm.interval(alpha=0.95, loc=p, scale=std)
# print(ci)

# 比例之差置信区间
import math
import scipy.stats as st
# n1, n2 = 500, 400
# p1, p2 = 225/500, 128/400
# p = p1 - p2
# std = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
# ci = st.norm.interval(alpha=0.95, loc=p, scale=std)
# print(ci)
# # 现代方法
# n1, n2 = 500 + 2, 400 + 2
# p1, p2 = (225+1)/n1, (128+1)/n2
# p = p1 - p2
# std = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
# ci = st.norm.interval(alpha=0.95, loc=p, scale=std)
# print(ci)

# 总体方差置信区间
# import numpy as np
# import pandas as pd
# import math
# from scipy import stats
# data = pd.read_excel('./食品重量.xlsx', header=None)
# test = np.array(data)
# import numpy as np
# from scipy import stats
# def confidence_interval_sigma(data, alpha):
#     tmp = (len(data) - 1) * np.var(data, ddof=1)
#     return (tmp / stats.chi2.ppf(1 - alpha / 2, df=len(data) - 1),
#                 tmp / stats.chi2.ppf(alpha / 2, df=len(data) - 1))
# print(confidence_interval_sigma(test, 0.05))
# # 方差比置信区间
# n1, n2 = 25 - 1, 25 - 1
# var1, var2 = 260, 280
# tmp = var1/var2
# F = stats.f(dfn=n1, dfd=n2)
# alpha = 0.1
# print(tmp / F.ppf(1 - alpha / 2), tmp / F.ppf(alpha / 2))

# # 求样本量
# import math
# import scipy.stats as st
# E, var = 400, 2000
# z = st.norm.ppf(1 - 0.05/2)
# print(math.floor((z**2 * var**2)/E**2)+1)
# var1, var2 = 90, 120
# E = 5
# print(math.floor((z**2 * (var1+var2))/E**2)+1)

# 估计总体比例样本量确定
import scipy.stats as st
import math
# pi, E = 0.9, 0.05
z = st.norm.ppf(1 - 0.05/2)
# print(math.floor((z**2 *pi*(1-pi))/E**2)+1)
pi1, pi2 = 0.5, 0.5
E = 0.1
print(math.floor((z**2 * (pi1*(1-pi1)+pi2*(1-pi2)))/E**2)+1)
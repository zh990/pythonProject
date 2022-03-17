# import math
# from scipy.stats import norm
# mean, mu = 330.5, 330
# s, n = 2, 50
# z = (mean-mu)/(s/math.sqrt(n))
# print(z)
# p = 2*(1-norm.cdf(z))
# print(p)

# import math
# import pandas as pd
# import numpy as np
# from scipy.stats import norm
# data = pd.read_excel('./PM25.xls', header=None)
# test = np.array(data)
# n = len(test)
# mean, std = np.mean(test), np.std(test, ddof=1)
# mu = 82
# z = (mean-mu)/(std/math.sqrt(n))
# print(z)
# p = norm.cdf(z)
# print(p)

# import math
# import pandas as pd
# import numpy as np
# from scipy.stats import t
# data = pd.read_excel('./demo9.xls')
# test = np.array(data)
# n = len(test)
# mean, std = np.mean(test), np.std(test, ddof=1)
# mu = 225
# t0 = (mean-mu)/(std/math.sqrt(n))
# print(t0)
# p = t.cdf(t0, n-1)
# print(p)

# import math
# from scipy.stats import norm
# z = (50-44)/math.sqrt(64/32+100/40)
# print(z)
# p = 2*(1-norm.cdf(z))
# print(p)

# # 均值之差t检验
# import pandas as pd
# import numpy as np
# import statsmodels.stats.weightstats as st
# from scipy.stats import t
# data = pd.read_excel('./demo9.xls')
# data1, data2 = np.array(data['机床甲'].dropna()), np.array(data['机床乙'].dropna())
# t0, p_two, df = st.ttest_ind(data1, data2)  # 方差相等
# print('t值：{}； p:{}'.format(t0, p_two))
# alpha = 0.05
# print('t的范围:', t.ppf(alpha/2, df), 1-t.ppf(alpha/2, df))
# t1, p_two_1, df1 = st.ttest_ind(data1, data2, usevar='unequal')  # 方差不相等
# print('t值：{}； p:{}'.format(t1, p_two_1))
# alpha = 0.05
# print('t的范围:', t.ppf(alpha/2, df), 1-t.ppf(alpha/2, df1))

# # 配对t检验
# import pandas as pd
# import numpy as np
# import scipy.stats as st
# data = pd.read_excel('./demo9.xls')
# data1, data2 = np.array(data['训练前'].dropna()), np.array(data['训练后'].dropna())
# t0, p = st.ttest_rel(data1, data2)
# print('t值：{}； p:{}'.format(t0, p))
# alpha, n = 0.05, len(data1)
# print('t的临界值:', st.t.ppf(1-alpha, n-1))

# # 总体比例检验
# import math
# import scipy.stats as st
# n = 10000
# p = 400/n
# pi0 = 0.02
# P = -(p-pi0)/math.sqrt(pi0*(1-pi0)/n)
# z = (st.norm.ppf(0.05/2), st.norm.ppf(1-0.05/2))
# print(P, z)

# # 两个总体比例之差为0检验
# import math
# import scipy.stats as st
# n1, n2 = 200, 200
# p1, p2 = 0.27, 0.35
# p = (p1*n1+p2*n2)/(n1+n2)
# z = (p1-p2)/math.sqrt(p*(1-p)*(1/n1+1/n2))
# print(z)
# p = st.norm.cdf(z)
# print(p)
# 两个总体比例之差为常数检验
# import math
# import scipy.stats as st
# n1, n2 = 150, 150
# p1, p2 = 68/150, 54/150
# d0 = 0.1
# z = ((p1-p2)-0.1)/math.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2)
# print(z)
# p = st.norm.cdf(z)
# print(p)
# 方差检验
# from scipy import stats
# std = 3.8
# n = 10
# sigma = 4
# Q = ((n-1)*(std**2))/sigma**2
# P = 1 - stats.chi2.cdf(Q, n-1)
# print(Q, P)

# 方差比检验
import pandas as pd
import numpy as np
import scipy.stats as st
data = pd.read_excel('./demo9.xls')
data1, data2 = np.array(data['供货商1'].dropna()), np.array(data['供货商2'].dropna())
n1, n2 = len(data1), len(data2)
var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
F = var1/var2
P = 2*(1-st.f.cdf(F, n1-1, n2-1))
print(F, P)


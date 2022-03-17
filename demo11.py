# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_excel('./贷款.xlsx')
# data0 = data['不良贷款'].values
# data1, data2, data3, data4 = data['贷款余额'].values, data['累计应收贷款'].values, \
#                              data['贷款项目个数'].values, data['固定资产投资额'].values
#
# fig = plt.figure()
# ax1 = plt.subplot(221)
# ax1.scatter(data1, data0)
# ax1.set_xlabel('贷款余额')
# ax1.set_ylabel('不良贷款')
# ax2 = plt.subplot(222)
# ax2.scatter(data2, data0)
# ax2.set_xlabel('累计应收贷款')
# ax2.set_ylabel('不良贷款')
# ax3 = plt.subplot(223)
# ax3.scatter(data3, data0)
# ax3.set_xlabel('贷款项目个数')
# ax3.set_ylabel('不良贷款')
# ax4 = plt.subplot(224)
# ax4.scatter(data4, data0)
# ax4.set_xlabel('固定资产投资额')
# ax4.set_ylabel('不良贷款')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.tight_layout()
# plt.show()
# # # 相关矩阵
# corr = data.corr()
# corr.to_excel('相关矩阵.xlsx')
# 相关性p值矩阵
# import numpy as np
# import math
# import scipy.stats as st
#
# def corrcoef_loop(r):
#     rows = r.shape[1]
#     df = rows**2-2
#     p = np.ones(shape=(rows, rows))
#     for i in range(rows):
#         for j in range(rows):
#             if i == j:
#                 p[i, j] = 0
#             else:
#                 p_ = 1 - st.t.cdf(abs(r[i, j])*math.sqrt(df/(1-r[i, j]**2)), df)
#                 p[i, j] = p[j, i] = p_
#     return p
# p = corrcoef_loop(np.array(corr))
# p = pd.DataFrame(p)
# p.columns = corr.columns
# p.index = corr.index
# p.to_excel('相关系数检验.xlsx')

# 一元线性回归
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_excel('./销售.xlsx')
# # 画散点图
# plt.scatter(data['广告费用'].values, data['销售量'].values)
# plt.xlabel('广告费用')
# plt.ylabel('销售量')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
# # 回归
from sklearn.linear_model import LinearRegression
x, y = data['广告费用'].values.reshape(-1, 1), data['销售量'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
# a = model.intercept_
# b = model.coef_
# print("最佳拟合线:截距", a, ",回归系数：", b)
# print('Y = '+str(b)+' X + '+ str(a))

# # 画出拟合曲线
# x2 = [[0], [30]]
# y2 = model.predict(x2)
# plt.plot(x2, y2, 'g-')
# yr = model.predict(x)
# for index, x in enumerate(x):
#     plt.plot([x, x], [y[index], yr[index]], 'r-')
# plt.show()
# 显著性检验
# from statsmodels.formula.api import ols
# model = ols('销售量~广告费用', data).fit()
# print(model.summary())

# 残差图
# def Residual_plot(x, y, model):
#     from matplotlib.font_manager import FontProperties
#     font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
#     y_prd = model.predict(x)
#     e = y - y_prd
#     sigama = np.std(e)
#     # 绘图
#     mx = max(x)[0] + 1
#     plt.scatter(x, e, c = 'red', s= 6)
#     plt.plot([0, mx], [2*sigama, 2*sigama], 'k--', c='green')
#     plt.plot([0, mx], [-2*sigama,- 2*sigama], 'k--', c='green')
#     plt.xlim(0, mx)
#     plt.ylim(-np.ceil(3*sigama+2), np.ceil(3*sigama+2))
#     plt.xlabel('x')
#     plt.ylabel('e')
#     plt.show()
#     return print(sigama)
# Residual_plot(x, y, model)

n = len(x)
y_prd = model.predict(x)
e = y - y_prd
sigama = np.std(e)
# zre = e / sigama  # ZRE 标准化残差
L_xx = n * np.var(x)
hii = 1/n + (x - np.mean(x))/L_xx   # 杠杆值
# sre = e/(sigama*np.sqrt(1 - hii))  # SRE 学生化残差
# print('标准化残差:', zre)
# print('学生化残差:', sre)
# 区间估计
from scipy.stats import t
import math
x0 = [[15]]
y0 = model.predict(x0)
h00 = 1/n + (15 - np.mean(x))/L_xx
lxr0, upr0 = y0-t.ppf(1-0.05/2, 10)*math.sqrt(h00)*sigama, y0+t.ppf(1-0.05/2, 10)*math.sqrt(h00)*sigama
lxr1, upr1 = y0-t.ppf(1-0.05/2, 10)*math.sqrt(1+h00)*sigama, y0+t.ppf(1-0.05/2, 10)*math.sqrt(1+h00)*sigama
print('点估计值：', y0)
print('新值区间估计：', lxr1, upr1)
print('新值平均值区间估计：', lxr0, upr0)

# 多元线性回归
import pandas as pd
import statsmodels.api as sm
import math
# data = pd.read_csv('./rock.csv')
# print(data.head())
# y = data['perm'].values
# X = data[['area', 'peri', 'shape']]
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())

# # 回归系数相关阵和协方差阵
# x = X2.values
# x01 = np.linalg.inv((np.dot(x.T, x)))
# y_prd = est2.predict(X2).values
# e = y - y_prd
# var = np.dot(e.T, e)/(48-4)
# covBeta = var*x01
# print('协方差阵：', covBeta)
# rows = covBeta.shape[1]
# p = np.ones(shape=(rows, rows))
# for i in range(rows):
#     for j in range(rows):
#         if i == j:
#             p[i, j] = 1
#         else:
#             p_ = covBeta[i, j]/(math.sqrt(covBeta[i, i])*math.sqrt(covBeta[j, j]))
#             p[i, j] = p[j, i] = p_
# print('相关阵：', p)

# # 二次线性回归
# X = data[['area', 'peri']]
# X3 = sm.add_constant(X)
# est = sm.OLS(y, X3)
# est3 = est.fit()
# y_prd = est3.predict(X3).values
# e = y - y_prd
# # print(est3.summary())
# # 区间估计
# from scipy.stats import t
# import math
# x0 = [[1, 7000, 2700]]
# y0 = est3.predict(x0)
# L_xx = np.dot(e.T, e)
# s = math.sqrt(L_xx/45)
# x = np.array(x0)
# d0 = np.dot(np.dot(x, np.linalg.inv(np.dot(X3.T, X3))), x.T)
# d0 = math.sqrt(d0)
# d1 = np.dot(np.dot(x, np.linalg.inv(np.dot(X3.T, X3))), x.T) + 1
# d1 = math.sqrt(d1)
# t = t.ppf(1-0.05/2, 45)
# L0, L1 = s*d0*t, s*d1*t
# lxr0, upr0 = y0-L0, y0+L0
# lxr1, upr1 = y0-L1, y0+L1
# print('点估计值：', y0)
# print('新值区间估计：', lxr1, upr1)
# print('新值平均值区间估计：', lxr0, upr0)

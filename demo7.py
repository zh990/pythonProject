import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
# x = np.arange(0, 21, 0.1)
# y1 = [st.chi2.pdf(i, 3) for i in x]
# y2 = [st.chi2.pdf(i, 5) for i in x]
# y3 = [st.chi2.pdf(i, 10) for i in x]
# for values, k in zip([y1, y2, y3], [3, 5, 10]):
#     plt.plot(x, values, label='chi2({})'.format(k))
# plt.legend(loc='best')
# plt.show()

# # 卡方分布计算概率
# p1 = st.chi2.cdf(10, 15)
# p2 = 1 - st.chi2.cdf(20, 15)
# x = st.chi2.ppf(0.95, 15)
# print(p1, p2, x)

# # 不同t分布和标准正态分布对比
# x = np.arange(-4, 4.1, 0.1)
# y1 = [st.t.pdf(i, 2) for i in x]
# y2 = [st.t.pdf(i, 5) for i in x]
# y3 = [st.norm.pdf(i, 0, 1) for i in x]
# for values, k in zip([y3, y1, y2], [(0, 1), 2, 5]):
#     if values == y3:
#         plt.plot(x, values, label='N{}'.format(k))
#     else:
#         plt.plot(x, values, linestyle='--', label='t({})'.format(k))
# plt.legend(loc='best')
# plt.show()

# # 求满足t分布的概率
# p1 = st.t.cdf(-2, 10)
# p2 = 1 - st.t.cdf(3, 10)
# x1, x2 = st.t.ppf(0.025, 10), st.t.ppf(0.975, 10)
# print(p1, p2, (x1, x2))

# # 不同F分布
# x = np.arange(0, 5.1, 0.02)
# y1 = [st.f.pdf(i, 10, 20) for i in x]
# y2 = [st.f.pdf(i, 5, 10) for i in x]
# y3 = [st.f.pdf(i, 3, 5) for i in x]
# for values, k in zip([y1, y2, y3], [(10, 20), (5, 10), (3, 5)]):
#     plt.plot(x, values, label='F{}'.format(k))
# plt.legend(loc='best')
# plt.show()

# # 求满足F分布的概率
# p1 = st.f.cdf(3, 10, 8)
# p2 = 1 - st.f.cdf(2.5, 10, 8)
# x = st.f.ppf(0.95, 10, 8)
# print(p1, p2, x)

# 样本均值和方差
import pandas as pd
# 整理数据
# data = []
# with open('./样本均值和方差.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         arr = line[:-1].split('\n\t')
#         for brr in arr:
#             sub_brr = brr[:-1].split(' ')
#         if sub_brr:
#             data.append(sub_brr)
# num = []
# for num_0 in data[1:]:
#     num_0 = list(map(float, num_0))
#     num.append(num_0)
# test = np.array(num)[:, 3]
# # 求样本均值和方差
# mean = np.mean(test)
# var = np.var(test)
# print(mean, var)
# # 总体分布图
# x = np.arange(2, 12, 2)
# y = pd.Series(x).value_counts()
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.bar(x=x, height=y)
# plt.xlabel('总体取值')
# plt.ylabel('频数')
# plt.show()
# # 样本均值分布直方图和密度曲线图
# import seaborn as sns
# sns.displot(data=test, kde=True, stat='probability', rug=True)
# plt.xticks(np.arange(min(test), max(test)+2, 2))
# plt.tight_layout()
# plt.show()

# 例题7.8
# import math
# mean = 60
# std = math.sqrt(6**2/50)
# print('样本均值服从N({},{:.2f}^2)'.format(mean, std))
# p = st.norm.cdf(57, mean, std)
# print(p)

# # 例题7.10
# import math
# pi = 0.02
# std = math.sqrt(pi*(1-pi)/600)
# p = st.norm.cdf(0.07, pi, std) - st.norm.cdf(0.025, pi, std)
# print(p)

# 例题7.11
import math
mean = 665 - 625
std = math.sqrt(20**2/8+25**2/8)
p = st.norm.cdf(0, mean, std)
print(p)





import numpy as np
import math
# s1 = np.arange(0, 5)
# s2 = np.array([0.75, 0.12, 0.08, 0.05])
# # 求均值
# mean = 0
# for i, j in zip(s1, s2):
#     mean += i * j
# print(mean)
# # 求方差
# var = 0
# for i, j in zip(s1, s2):
#     var += ((i - mean)**2) * j
# print(var)
# print(math.sqrt(var))

# # 标准差对比
# # 整理数据
# data = []
# with open('./标准差对比.txt', 'r', encoding='utf-8') as f:
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
# num_A, num_B = np.array(num[:7]), np.array(num[7:])
# # 求标准差
# var_A, var_B = 0, 0
# for i, j in zip(num_A[:, 0], num_A[:, 1]):
#     var_A += ((i - 7)**2) * j
# for i, j in zip(num_B[:, 0], num_B[:, 1]):
#     var_B += ((i - 7)**2) * j
# print(math.sqrt(var_A), math.sqrt(var_B))
#
# 离散分布 pmf:该点概率值 cdf:P(X<=x) ppf:已知累积概率求对应点
# 连续分布 pdf:取该点附近值 cdf:P(X<=x) ppf:已知累积概率求对应点
# # 二项分布求概率
# from scipy.stats import binom, poisson
# # 没有次品的概率
# print(binom.pmf(0, 5, 0.06))
# # 恰好有一个次品的概率
# print(binom.pmf(1, 5, 0.06))
# # 有三个及三个以下的次品概率
# print(binom.cdf(3, 5, 0.06))
#
# # 泊松分布求概率
# print(poisson.pmf(5, 2.5))

# 绘制均值不同，方差相同（N（-2，1）、N（2，1））的正态分布曲线
import scipy.stats as st
import matplotlib.pyplot as plt
#
# x1 = np.arange(-2, 6.1, 0.1)
# x2 = np.arange(-6, 2.1, 0.1)
# y1 = st.norm.pdf(x1, 2, 1)
# y2 = st.norm.pdf(x2, -2, 1)
# plt.plot(x1, y1, label='N(2,1)')
# plt.plot(x2, y2, label='N(-2,1)')
# plt.legend(loc='best')
# plt.show()

# 绘制均值相同，方差不同（N（0，0.5）、N（0，1）、N（0，2））的正态分布曲线
# x = np.arange(-3, 3.1, 0.1)
# y1 = st.norm.pdf(x, 0, 0.5)
# y2 = st.norm.pdf(x, 0, 1)
# y3 = st.norm.pdf(x, 0, 2)
# plt.plot(x, y1, label='N(0,0.5)')
# plt.plot(x, y2, label='N(0,1)')
# plt.plot(x, y3, label='N(0,2)')
# plt.legend(loc='best')
# plt.show()

# 正态分布概率计算
# p0 = st.norm.cdf(40, 50, 10)
# p1 = p0 - st.norm.cdf(30, 50, 10)
# print(p0, p1)
# p2 = st.norm.cdf(2.5, 0, 1)
# p3 = st.norm.cdf(2, 0, 1) - st.norm.cdf(-1.5, 0, 1)
# p4 = st.norm.ppf(0.025, 0, 1)
# print(p2, p3, p4)

# 正态分布近似求概率
# p0 = st.norm.cdf(86, 80, 4) - st.norm.cdf(70, 80, 4)
# p1 = 1 - st.norm.cdf(80, 80, 4)
# print(p0, p1)

# 绘制PP图和QQ图
# 整理数据
data = []
with open('./正态分布.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        arr = line[:-1].split('\n\t')
        for brr in arr:
            sub_brr = brr[:-1].split(' ')
        if sub_brr:
            data.append(sub_brr)
num = []
for num_0 in data[1:]:
    num_0 = list(map(float, num_0))
    num.append(num_0)
final_data = np.array(num)[:, 2]
data_sort = np.array(sorted(final_data))
n = len(data_sort)
# PP图
x1 = np.arange(1/(2*n), 1, 1/n)
y1 = st.norm.cdf(data_sort, data_sort.mean(), data_sort.std())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot([0, 1], [0, 1], color='r')
plt.scatter(x1, y1)
plt.xlabel('观测累积概率')
plt.ylabel('期望累积概率')
plt.title('P-P plot')
plt.show()

# QQ图
# fig = plt.figure()
# res = st.probplot(data_sort, plot=plt)
# plt.title('Q-Q plot')
# plt.show()

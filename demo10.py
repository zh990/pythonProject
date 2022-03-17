import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# 描述统计
# df = pd.read_excel('./demo10.xlsx')
# data1, data2, data3 = np.array(df['品种1'].dropna()), np.array(df['品种2'].dropna()),\
#                        np.array(df['品种3'].dropna())
# data1, data2, data3 = data1.tolist(), data2.tolist(), data3.tolist()
# list_groups = [data1, data2, data3]
# list_total = data1+data2+data3
# print(df.describe())
# df.plot.box()
# plt.xticks(rotation=45)
# plt.ylabel('产量')
# plt.grid(linestyle="--", alpha=0.8)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
# # 方差分析
# F, p = st.f_oneway(data1, data2, data3)
# print(F, p)
# # 均值图
# df1 = {'品种1':[df['品种1'].min(), df['品种1'].mean(), df['品种1'].max()],
#        '品种2':[df['品种2'].min(), df['品种2'].mean(), df['品种2'].max()],
#        '品种3':[df['品种3'].min(), df['品种3'].mean(), df['品种3'].max()]}
# df1 = pd.DataFrame(df1, index=['min', 'mean', 'max'])
# df1.plot(linestyle='dashed', marker='o')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
# LSD多重比较
# import math
# import numpy as np
# import scipy.stats as st
# def SE(group):
#     se=0
#     mean1=np.mean(group)
#     for i in group:
#         error=i-mean1
#         se+=error**2
#     return se
# def SSE(list_groups):
#     sse=0
#     for group in list_groups:
#         se=SE(group)
#         sse+=se
#     return sse
# def MSE(list_groups,list_total):
#     sse=SSE(list_groups)
#     mse=sse/(len(list_total)-1*len(list_groups))*1.0
#     return mse
# def LSD(list_groups,list_total,sample1,sample2):
#     mean1=np.mean(sample1)
#     mean2=np.mean(sample2)
#     distance=abs(mean1-mean2)
#     # t检验的自由度
#     df=len(list_total)-1*len(list_groups)
#     mse=MSE(list_groups,list_total)
#     a = 0.05
#     t_value=st.t(df).isf(a/2)
#     lsd=t_value*math.sqrt(mse*(1.0/len(sample1)+1.0/len(sample2)))
#     print('lsd:',lsd)
#     t = (mean1-mean2-lsd, mean1-mean2+lsd)
#     print('品种1与品种3均值之差的置信区间：{}'.format(t))
#     if distance<lsd:
#         print('在置信度95%下，品种1与品种3无显著差异')
#     else:
#         print('在置信度95%下，品种1与品种3有显著差异')
#     return 'LSD多重比较'
#
# lsd = LSD(list_groups, list_total, data1, data3)
# print(lsd)

# # 效应分析
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# data_melt = df.melt()
# data_melt.columns = ['品种', '产量']
# anova_re = anova_lm(ols('产量~品种', data=data_melt).fit(), type=2)
# print(pd.DataFrame(anova_re))
# # HSD多重比较
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# print(pairwise_tukeyhsd(data_melt['产量'], data_melt['品种']))
# # 画置信区间图
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(1, 4)
# y = [-10, -2, 8]
# err = [-5.2867, -5.2867, 5.2867]
# x_ticks = ("品种1-2", "品种1-3", "品种2-3")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.errorbar(x, y, yerr=err, color="black", capsize=3,
#              linestyle="None",
#              marker="s", markersize=7, mfc="black", mec="black")
# plt.xticks(x, x_ticks, rotation=45)
# plt.tight_layout()
# plt.show()

# 多因素方差分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data = pd.read_excel('./demo10.xlsx', skiprows=11)
# data.groupby('施肥方式').plot.box()
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()
# 描述统计
# data1 = data.groupby('施肥方式')
# data_miaoshu = data1.describe()
# data_miaoshu.to_excel('描述统计.xlsx')
# print(data1.describe())
# print('mean:', data1.mean())
# print('std:', data1.std())

# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# data2 = pd.read_excel('./demo101.xlsx')
# 双因素方差分析
# model = ols('产量 ~施肥方式 + 品种', data2).fit()
# result = pd.DataFrame(anova_lm(model))
# print(result)
# sf, pz = result.at['施肥方式', 'sum_sq'], result.at['品种', 'sum_sq']
# zong = result['sum_sq'].sum()
# eta_sq0, eta_sq1 = sf/zong, pz/zong
# print('施肥方式主效应：', eta_sq0, '品种主效应：', eta_sq1)
# eta_sq_part0, eta_sq_part1 = sf/(zong-pz), pz/(zong-sf)
# print('施肥方式偏效应：', eta_sq_part0, '品种偏效应：', eta_sq_part1)
# 交互效应方差分析
# model = ols('产量 ~施肥方式 + 品种 + 施肥方式:品种', data=data2).fit()
# anova_table = anova_lm(model, type=2)
# result = pd.DataFrame(anova_table)
# sf, pz, p, wucha= result.at['施肥方式', 'sum_sq'], result.at['品种', 'sum_sq'],\
#                   result.at['施肥方式:品种', 'sum_sq'], result.at['Residual', 'sum_sq']
# eta_sq_part0, eta_sq_part1, eta_sq_part2 = sf/(sf+wucha), pz/(pz+wucha), p/(p+wucha)
# print('施肥方式偏效应：', eta_sq_part0, '品种偏效应：', eta_sq_part1, '交互效应偏效应：', eta_sq_part2)

# # 有无交互效应的折线图和箱线图
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# data2 = pd.read_excel('./demo101.xlsx')
# data20 = data2.groupby(['施肥方式', '品种']).产量
# pz0 = list(data20.mean())
# data21 = data2.groupby('施肥方式')
# for key, value in data21:
#     if key == '甲':
#         t0 = value
#         t0.index = np.arange(0, 15)
#     else:
#         t1 = value
#         t1.index = np.arange(0, 15)
# sf0 = pd.DataFrame({'甲': t0['产量'], '乙': t1['产量']})
# data22 = data2.groupby('品种').产量
# for key, value in data22:
#     if key == '品种1':
#         t2 = value.values.tolist()
#     elif key == '品种2':
#         t3 = value.values.tolist()
#     else:
#         t4 = value.values.tolist()
# data23 = data2.groupby(['品种', '施肥方式']).产量
# for key, value in data23:
#     print(key)
#     print(value)
# sf1 = list(data23.mean())
# print(sf1)
# fig = plt.figure()
# plt.rcParams['font.sans-serif'] = ['SimHei']
# ax1 = plt.subplot(221)
# ax1.plot(np.arange(1, 4), pz0[0:3])
# ax1.plot(np.arange(1, 4), pz0[3:])
# ax1.set_ylabel('产量')
# ax2 = plt.subplot(222, sharey=ax1)
# ax2.boxplot(sf0, labels=['甲', '乙'])
# ax3 = plt.subplot(223, sharex=ax1)
# ax3.boxplot([t2, t3, t4])
# ax3.set_xlabel('品种')
# ax3.set_ylabel('产量')
# ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
# ax4.plot(np.arange(1, 3), sf1[1::-1])
# ax4.plot(np.arange(1, 3), sf1[3:1:-1])
# ax4.plot(np.arange(1, 3), sf1[:3:-1])
# ax4.set_xlabel('施肥方式')
# plt.show()

# 正态性检验Q-Q图
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
# df = pd.read_excel('./demo10.xlsx', nrows=10)
# data1, data2, data3 = df['品种1'].values, df['品种2'].values, df['品种3'].values
# data1, data2, data3 = data1.tolist(), data2.tolist(), data3.tolist()
# data1, data2, data3 = sorted(data1), sorted(data2), sorted(data3)
# fig = plt.figure()
# ax1 = plt.subplot(131)
# st.probplot(data1, plot=ax1)
# ax2 = plt.subplot(132)
# st.probplot(data2, plot=ax2)
# ax3 = plt.subplot(133)
# st.probplot(data3, plot=ax3)
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# ax1.set_title('品种1 Q-Q plot')
# ax2.set_title('品种2 Q-Q plot')
# ax3.set_title('品种3 Q-Q plot')
# plt.show()
# data2 = pd.read_excel('./demo101.xlsx')
# data_sort = sorted(data2['产量'].values)
# res = st.probplot(data_sort, plot=plt)
# plt.title('Q-Q plot')
# plt.show()

# # KS检验
# import pandas as pd
# import scipy.stats as st
# df = pd.read_excel('./demo101.xlsx')
# u = df['产量'].mean()
# std = df['产量'].std()
# stat_KS, p_KS = st.kstest(df['产量'], 'norm', (u, std))
# print('stat_KS=%f, p_KS=%f' % (stat_KS, p_KS))
# # SK检验
# from scipy.stats import shapiro
# stat, p = shapiro(df['产量'].values)
# print('stat_SK=%f, p_SK=%f' % (stat, p))

# # 残差图!!!! 还有个多重比较
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# df = pd.read_excel('./demo10.xlsx', nrows=10)
# data1, data2, data3 = np.array(df['品种1']), np.array(df['品种2']), np.array(df['品种3'])
# mean = (sum(data1)+sum(data2)+sum(data3))/(len(data1)+len(data2)+len(data3))
# erro1, erro2, erro3 = data1-mean, data2-mean, data3-mean
# print(erro1, erro2, erro3)
# erro0 = np.append(erro1, erro2, axis=0)
# erro = np.append(erro0, erro3, axis=0)
# print(erro)
# pz = np.ones_like(erro)
# pz[10:20] = pz[10:20] + 1
# pz[20:29] = pz[20:29] + 2
# print(pz)
# plt.scatter(x=pz, y=erro)
# plt.show()

# 方差齐性检验:Barlett检验
from scipy.stats import bartlett
import pandas as pd
data = pd.read_excel('./demo101.xlsx')
data1 = data.groupby('施肥方式')
for key, value in data1:
    if key == '甲':
        t0 = value
    else:
        t1 = value
stat1, p1 = bartlett(t0['产量'].values, t1['产量'].values)
data2 = data.groupby('品种')
for key, value in data2:
    if key == '品种1':
        t2 = value
    elif key == '品种2':
        t3 = value
    else:
        t4 = value
# stat2, p2 = bartlett(t2['产量'].values, t3['产量'].values, t4['产量'].values)
# print(stat1, p1)
# print(stat2, p2)
#
# # 方差齐性检验:Levene检验
# from scipy.stats import levene
# stat3, p3 = levene(t0['产量'].values, t1['产量'].values)
# stat4, p4 = levene(t2['产量'].values, t3['产量'].values, t4['产量'].values)
# print(stat3, p3)
# print(stat4, p4)
# 单因素方差分析Kruskal-Wallis检验
import scipy.stats as stats
H, P = stats.kruskal(t2['产量'].values, t3['产量'].values, t4['产量'].values)
print(H, P)







import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 绘制柱状图
test = pd.read_excel("./牙膏.xlsx")
test_array = np.array(test)
x = test_array[:, 0]
y = test_array[:, 1]
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.bar(x=x, height=y)
# plt.xlabel('牙膏')
# plt.ylabel('频数')
# plt.show()
#
# #绘制复式柱状图
# x_label = test_array[:, 0]
# group_A = test_array[:, 1]
# group_B = test_array[:, 2]
# plt.rcParams['font.sans-serif'] = ['SimHei']
# width = 0.3
# x = np.arange(1, 7)
# x_0 = x - width
# plt.bar(x=x_0, height=group_A, width=width, color='green', tick_label=x_label, label='A_group')
# plt.bar(x=x, height=group_B, width=width, color='blue', tick_label=x_label, label='B_group')
# plt.xlabel('牙膏')
# plt.ylabel('频数')
# plt.legend(loc='best')
# plt.show()
#
# # 绘制饼图
# plt.pie(y, labels=x, autopct='%1.1f%%')
# plt.show()
#
# # 绘制直方图和密度曲线
import seaborn as sns
test = pd.read_excel('.\成绩表.xlsx', index_col="序号")
test_array = np.array(test)
shufen = test_array[:, 1]
# sns.displot(data=shufen, kde=True, stat='density')
# plt.tight_layout()
# plt.show()

# # 绘制茎叶图，没有直接的包可使用
# from itertools import groupby
# for k, g in groupby(sorted(shufen), key=lambda x: int(x) // 10):
#     lst = map(str, [int(y) % 10 for y in list(g)])
#     print (k, '|', ' '.join(lst))

# # 绘制箱线图
# plt.boxplot(shufen, vert=False)
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()

# # 绘制雷达图（不同地区牙膏对比），没有直接的包可使用
# z = test_array[:, 2]
# # 设置每个数据点的显示位置，在雷达图上用角度表示
# angles = np.linspace(0, 2*np.pi, len(x), endpoint=False)
# angles = np.concatenate((angles, [angles[0]]))
# # 绘图
# fig = plt.figure()
# for values in [y, z]:
#     # 拼接数据首尾，使图形中线条封闭
#     values = np.concatenate((values, [values[0]]))
#     # 设置为极坐标格式
#     ax = fig.add_subplot(111, polar=True)
#     # 绘制折线图
#     ax.plot(angles, values, 'o-', linewidth=2)
#     # 填充颜色
#     ax.fill(angles, values, alpha=0.25)
#     # 设置图标上的角度划分刻度，为每个数据点处添加标签
#     ax.set_thetagrids(angles[0:6] * 180 / np.pi, x)
#
#     # 设置雷达图的范围
#     ax.set_rlim(0, 30)
# # 添加标题
# plt.title('不同地区消费者购买牙膏情况')
# plt.legend(["A地区", "B地区"], loc='best')
# # 添加网格线
# ax.grid(True)
# plt.show()
# 方法2
# angles = np.linspace(0, 2*np.pi, len(group_A), endpoint=False)
# angles = np.append(angles, 0)
# # 遍历绘制两组雷达图
# for values in [group_A, group_B]:
#     r = np.append(values, values[0])
#     plt.polar(angles, r, 'o-', lw=2)
#     plt.thetagrids(angles[0:6]*180/np.pi, x_label, fontproperties='simhei')
# plt.ylim(0, 30)
# plt.legend(["A地区", "B地区"], loc='best')
# plt.show()
# plt.plot

# 绘制数学分析和高等代数散点图
gaodai = test_array[:, 2]
plt.scatter(shufen, gaodai)
plt.xlabel('数学分析')
plt.ylabel('高等代数')
plt.show()


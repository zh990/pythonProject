import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

# 整理数据
data = []
with open('./描述统计.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        arr = line[:-1].split('\n\t')
        for brr in arr:
            sub_brr = brr[:-1].split('  ')
        if sub_brr:
            data.append(sub_brr)
num = []
for num_0 in data[1:]:
    num_0 = list(map(float, num_0))
    num.append(num_0)
df = pd.DataFrame(num, columns=data[0])
# 描述统计
sum = df.sum(axis=0)
mean = df.mean(axis=0)
var = df.var(axis=0)
skew = df.skew(axis=0)
kurt = df.kurt(axis=0)
summary = pd.concat([sum, mean, var, skew, kurt], join='outer', axis=1)
summary.columns = ['sum', 'mean', 'var', 'skew', 'kurt']
print(summary)
#  绘制多组箱线图
print(df.describe())  # 显示总个数，平均值，最大/小值，中位数、上下四分位数、标准偏差等内容
df.plot.box()
plt.xticks(rotation=45)
plt.grid(linestyle="--", alpha=0.8)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.show()

df.plot(title='运动员成绩折线图')
plt.xticks(np.arange(0, 25, 5))
plt.ylabel('运动员成绩')
plt.show()





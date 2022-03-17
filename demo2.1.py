import pandas as pd
test = pd.read_excel('.\成绩表.xlsx', index_col="序号")
# shufen = test.sort_values(by='数分', axis=0, ascending=False).iloc[0:3]
# print(shufen)
# test1 = test[(test['数分'] > 70) &
#              (test['高代'] > 70) &
#              (test['概率论'] > 70) &
#              (test['C++'] > 70) &
#              (test['CET-4'] > 70) &
#              (test['颜值'] < 100)]
# print(test1)
test1 = test[(test > 70)].dropna()
test2 = test1[test1['颜值'] < 100]
print(test2)
# yanzhi = test.sort_values(by='颜值', axis=0, ascending=False)
# print(yanzhi)
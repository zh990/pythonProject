import pandas as pd
import numpy as np
test = pd.read_excel('./统计表和统计图.xlsx')
a = test['饮料类别'].value_counts()
b = test['性别'].value_counts()
test['value'] = np.arange(len(test['饮料类别']))
c = test.groupby(['性别', '饮料类别']).count()
print(a)
print(b)
print(c)


import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

np.random.seed(114514)
cls1 = DataFrame({'x1':np.random.normal(105,5**2,100),\
'x2':np.random.normal(2800,4**2,100),\
'cls':[1 for i in range(100)]},columns=['x1','x2','cls'])
cls2 = DataFrame({'x1':np.random.normal(115,5**2,100),\
'x2':np.random.normal(2700,4**2,100),\
'cls':[-1 for i in range(100)]},columns=['x1','x2','cls'])

df = pd.concat([cls1,cls2])

plt.scatter(cls1['x1'],cls1['x2'],color='r')
plt.scatter(cls2['x1'],cls2['x2'],color='b')
plt.show()

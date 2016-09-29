import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import time

#データの正規化
def scale_data(data):
	sc_data = data
	sc_data.x1 = scale(data.x1)
	sc_data.x2 = scale(data.x2)
	return sc_data

#Ldual(a)の微分
def d_Ldual(xx,yy,a,x,y):
	ret = .0
	for aaa,xxx,yyy in zip(a,x,y):
		ret += aaa*yy*yyy*np.dot(xx,xxx)
	return 1-ret

#最急降下法
def SDM(a,data,eta=0.01):
	change = 1
	y = np.array(data.cls)
	x = np.array(pd.concat([data.x1,data.x2],axis=1))
	while change > 0.5:
		change=0
		new_a = []
		for aa,xx,yy in zip(a,x,y):
			if aa != 0:
				dif = d_Ldual(xx,yy,a,x,y)
				new_a.append(max(0,aa+eta*dif))
				change += dif**2
			else:
				new_a.append(0)
		a = new_a
		#print(a,change)
		#time.sleep(0.1)
	return np.array(a)

#未定乗数を用いて重みを算出
def calc_weight(a,data):
	y = np.array(data.cls)
	x = np.array(pd.concat([data.x1,data.x2],axis=1))
	w = np.array([.0,.0])
	S_len = len(a[a!=0])
	w0 = 0
	for aa,yy,xx in zip(a,y,x):
		w += aa*yy*xx
		if aa != 0:
			tmp = 0
			for aaa,yyy,xxx in zip(a,y,x):
				if aa != 0:
					tmp += aaa*yyy*np.dot(xx,xxx)
			w0 += yy - tmp
	w0 /= S_len
	return(w,w0)

print("read data.")
df = pd.read_csv("fishDataEasy_train.csv")
#print(df)
print("preprocessing")
scale_df = scale_data(df.copy())
a = np.ones(len(scale_df))
print("SDM")
a = SDM(a,scale_df)
print("calc_weight")
w,w0 = calc_weight(a,scale_df)
print(w,w0)

x_fig = np.linspace(scale_df.x1.min()-0.5,scale_df.x1.max()+0.5,100)
y_fig = -(w[0]/w[1])*x_fig -(w0/w[1])
plt.scatter(scale_df[scale_df['cls']==1].x1,scale_df[scale_df['cls']==1].x2,color='r')
plt.scatter(scale_df[scale_df['cls']==-1].x1,scale_df[scale_df['cls']==-1].x2,color='b')
plt.plot(x_fig,y_fig)
plt.show()

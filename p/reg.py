'''
Created on 2020/04/22

@author: Kyoko
'''
import pandas as pd

df = pd.read_csv('data.csv')
df.head()
x = df[['day']]
y = df[['Num']]

print(x)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression

model_lr = LinearRegression()
model_lr.fit(x, y)

import matplotlib.pyplot as plt

plt.plot(x, y, 'o')
#plt.plot(x, model_lr.predict(x), linestyle="solid")
#plt.show()

print('1.モデル関数の回帰変数 w1: %.3f' %model_lr.coef_)
print('1.モデル関数の切片 w2: %.3f' %model_lr.intercept_)
print('1.y= %.3fx + %.3f' % (model_lr.coef_ , model_lr.intercept_))
print('1.決定係数 R^2： ', model_lr.score(x, y))

df = pd.read_csv('taeget.csv')
df.head()
x2 = df[['day']]
#plt.plot(x2, model_lr.predict(x2), linestyle="solid")
plt.show()

# print(model_lr.predict(x))
#
#
# model_lr = ARDRegression()
# model_lr.fit(x, y)

#
# print('2.モデル関数の回帰変数 w1: %.3f' %model_lr.coef_)
# print('2.モデル関数の切片 w2: %.3f' %model_lr.intercept_)
# print('2.y= %.3fx + %.3f' % (model_lr.coef_ , model_lr.intercept_))
# print('2.決定係数 R^2： ', model_lr.score(x, y))
#
# df = pd.read_csv('taeget.csv')
# df.head()
# x2 = df[['day']]
# plt.plot(x2, model_lr.predict(x2), linestyle="solid")
# plt.show()
#
# print(model_lr.predict(x))

import numpy as np
import pandas as pd
from imputation import Impute
from missing_mat import MissMat
import statsmodels.api as sm

TEST = 700
filename = 'FRED-MD.csv'

def get_regression_data(column_y, t_end=733, method=0, r=3):
    data = pd.read_csv(filename)
    y = data[column_y].to_numpy()[:t_end]
    x = data.drop(['Date', column_y], axis=1).to_numpy()[:t_end]
    w = np.ones(x.shape)
    w[np.isnan(x)] = 0
    mdata = MissMat(x, w)
    mc = Impute(mdata)
    if method == 0:
        _ = mc.amputation(r)
        F = mc.F
    if method == 1:
        _ = mc.fit_via_TP(r)
        F = mc.F
    if method == 2:
        _ = mc.fit_via_Weight(r)
        F = mc.F
    if method == 3:
        _ = mc.fit_via_Nuclear(r)
        F = mc.F
    if method == 4:
        _ = mc.fit_via_PQ(r)
        F = mc.F
    return y, F


def one_step_predict(column_y, step=0, lags=4, method=0):
    y, F = get_regression_data(column_y, t_end=TEST+step, method=method)
    data = pd.DataFrame(F, columns=['f1','f2','f3'])
    data['y'] = y
    cols = ['y', 'f1','f2','f3']
    data = data[cols]
    for l in range(lags):
        data = pd.concat([data, data['y'].shift(l+1).rename('y_{}'.format(l+1))], axis=1)
        data = pd.concat([data, data['f1'].shift(l+1).rename('f1_{}'.format(l+1))], axis=1)
        data = pd.concat([data, data['f2'].shift(l+1).rename('f2_{}'.format(l+1))], axis=1)
        data = pd.concat([data, data['f3'].shift(l+1).rename('f3_{}'.format(l+1))], axis=1)
    data = data.iloc[lags:]
    model = sm.OLS(data['y'],data.iloc[:,4:])
    results = model.fit()
    yhat = results.predict(data.iloc[-1:,:len(data.columns)-4].to_numpy())
    return yhat[0]


def out_sample_mse(column_y, method=0):
    yhats = []
    for t in range(733-TEST):
        yhats.append(one_step_predict(column_y, step=t, method=method))
    yhats = np.array(yhats)
    data = pd.read_csv(filename)
    y = data[column_y].to_numpy()[TEST:]
    rmse = np.mean((y - yhats)**2)
    return rmse, yhats

"""
cols = ['INDPRO', 'UNRATE', 'CPIAUCSL']
y, F = get_regression_data(cols[0], method=0)
rmse = out_sample_mse(cols[2])
print(rmse)
rmse = out_sample_mse(cols[2], method=1)
print(rmse)
rmse = out_sample_mse(cols[2], method=2)
print(rmse)
rmse = out_sample_mse(cols[2], method=3)
print(rmse)
"""

import matplotlib.pyplot as plt

cols = ['INDPRO', 'UNRATE', 'CPIAUCSL']
rmse, yhats = out_sample_mse(cols[0])
df = pd.read_csv(filename)
y = df[cols[0]].to_numpy()
yhat = np.copy(y)
yhat[700:] = yhats
plt.figure(figsize=(10,5))
plt.plot(y[630:], color='blue', label='Real IP Index')
plt.plot(yhat[630:], color='black', ls='-', lw=2,label='1 step forecast of IP Index')
plt.legend()


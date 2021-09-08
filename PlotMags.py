#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import sys
import scipy
import scipy.optimize as optimize
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

def runRANSAC(df, maglimit, iterations):
    X = []
    y = []
    Gx = []
    Gy = []

    for i in range(len(df.index)):
        if df.loc[i,'GMAG'] > 0 and df.loc[i,'GMAG'] < maglimit:
            #print df.loc[i,'GMAG']
            X.append(df.loc[i,'GMAG'])
            y.append(df.loc[i,'MAG'])
            Gx.append(df.loc[i,'RA'])
            Gy.append(df.loc[i,'DEC'])

    X = np.reshape(X,(-1,1))
    y = np.reshape(y,(-1,1))

    line_y_ransac_tmp = []
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]

    for i in range(iterations):
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(residual_threshold=0.5, stop_probability=0.9999)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(line_X)
        line_y_ransac_tmp.append(line_y_ransac)

    line_y_ransac = np.mean(line_y_ransac_tmp,axis=0)

    return X, y, line_y_ransac, line_X, inlier_mask, outlier_mask
    # Predict data of estimated models

if len(sys.argv) > 1:
    pklfile = sys.argv[1]
else:
    pklfile = 'out.pkl'

df = pd.read_pickle(pklfile)

df = df[df.MAG != 99.0]

df['BmG'] = df.BMAG - df.GMAG
df['BmR'] = df.BMAG - df.RMAG
df['GmR'] = df.GMAG - df.RMAG

# df = df[(df.MAG < 0) & (df.GMAG < 14)]
df = df.reset_index(drop=True)

df.to_csv('./file.txt', sep=' ')

BmR = df.BMAG - df.RMAG
BmG = df.BMAG - df.GMAG
GmR = df.GMAG - df.RMAG


# imag = np.array([10.173, 11.649, 11.154, 11.819, 10.779, 11.02, 9.882, 10.161, 12.119, 12.969, 12.559, 12.655])
# pmag = np.array([11.721, 13.270, 13.231, 14.281, 13.833, 13.347, 12.362, 12.750, 17.230, 16.730, 14.673, 14.681])
# imag = 10.173
# pmag = 11.721
# ci = np.array([0.458, 0.481, 0.804, 1.096, 1.484, 0.932, 1.187, 1.208, 2.394, 1.770, 0.831, 0.748])
# ci = 0.458

# imag = -1*df['MAG'][0:10].to_numpy()
# pmag = df['GMAG'][0:10].to_numpy()
# ci = df['GmR'][0:10].to_numpy()

# print(ci)
# print(type(ci))

# def magcalc(params):
#     tprime, zprime = params
#     cmag = np.add(imag,tprime*ci) + zprime # cmag = calculated mag, ci = colour index
#     return cmag

# def con(params):
#     cmag = magcalc(params)
#     residual = np.subtract(pmag,cmag)
#     # print(np.mean(residual))
#     return np.mean(residual)

# def res(params, imag, ci, pmag):
#     cmag = magcalc(params)
#     residual = np.subtract(pmag,cmag) # pmag = photometric mag
#     stddev = np.std(residual)
#     return stddev

# cons = {'type':'eq', 'fun': con}

# result = optimize.minimize(res, (1.5,0.8), method='Powell', constraints=cons, bounds=((0,2),(0,2)), args=(imag,ci,pmag))

# print(result)

# print(ci[0], imag[0], pmag[0])

# newmag = imag + result.x[0]*ci + result.x[1]
# print(newmag)
# print(pmag)

# print(newmag - pmag)



#for i in range(len(df.index)):
#    print(df.loc[i,'FLUX'], df.loc[i,'MAG'], df.loc[i,'GMAG'])

X, y, line_y_ransac, line_X, inlier_mask, outlier_mask = runRANSAC(df, 12, 100)
X15, y15, line_y_ransac15, line_X15, inlier_mask15, outlier_mask15 = runRANSAC(df, 15, 100)


mR = (line_y_ransac[len(line_y_ransac)-1]-line_y_ransac[0]) / (line_X[len(line_X)-1]-line_X[0])
bR = line_y_ransac[0]-mR*line_X[0]
# print(mR, bR)

b = y-X
# print(np.mean(b))

x2 = np.linspace(0,15,100)
y2 = mR*x2 + bR

#plt.scatter(Gx, Gy, color='r', alpha=0.3)



plt.figure(figsize=(14,10))

# plt.scatter(df['GMAG'],df['MAG'], marker='o', color='black')

plt.scatter(df.GMAG, df.MAG, color='black', marker='o', label='All Data')
# plt.scatter(df.RMAG, df.MAG, color='red', marker='o', label='All Data')
# plt.scatter(df.BMAG, df.MAG, color='blue', marker='o', label='All Data')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='o', edgecolors='black', label='RANSAC Outliers')
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='o', edgecolors='black', label='RANSAC Inliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2, label='y={0:0.2f}'.format(mR[0]) + '*' + r'M$_\mathrm{G}$' + ' + {0:0.2f}'.format(bR[0]) + ' (RANSAC)')
plt.plot(x2,y2)
plt.xlim(6,12)
plt.ylim(-10,0)
# plt.ylim(0,0.00000000001)
# plt.title('Instrumental Magnitude vs GAIA DR2 G Magnitude')
plt.xlabel('GAIA G-band Magnitude', size=15)
plt.ylabel('Instrumental Magnitude', size=15)
plt.legend()

plt.tight_layout()
plt.savefig('IMag-vs-GMag-EMCCD.png', dpi=300)

plt.show()

# plt.figure(figsize=(14,10))

# plt.scatter(df.BmR, df.MAG, color='blue', marker='o', label='All Data')
# plt.scatter(GmR, df.MAG, color='red', marker='o', label='All Data')
# plt.scatter(BmG, df.MAG, color='green', marker='o', label='All Data')

# plt.show()

# In[ ]:





# In[ ]:





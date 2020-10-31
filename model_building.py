#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:47:13 2020

@author: pierreperrin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

#choose relevant columns
df.columns


df_model = df[['avg_salary','Rating', 'Size','Type of ownership','Industry','Sector','Revenue','job_state','age','python_ym','spark','aws','excel','job_simp','seniority','desc_len']]


#get dummy data
df_dum = pd.get_dummies(df_model)

#train text split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear model
import statsmodels.api as sm
X_sm = sm.add_constant(X)
model = sm.OLS(y,X_sm)
#print(model.fit().summary())

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

ln = LinearRegression()
ln.fit(X_train, y_train)

np.mean(cross_val_score(ln,X_train,y_train, scoring='neg_mean_absolute_error', cv=3))


#lasso regression
ln_l = Lasso()
np.mean(cross_val_score(ln_l,X_train,y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/10)
    lnl = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lnl,X_train,y_train, scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

#random forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train, scoring='neg_mean_absolute_error', cv=3))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('msi','mae'), 'max_features':('auto','sqrt','log2')}


gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error', cv=3 )
gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

#test ensembles
tpred_ln = ln.predict(X_test)
tpred_lnl = ln_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_ln)
mean_absolute_error(y_test, tpred_lnl)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (tpred_ln+tpred_rf)/2)





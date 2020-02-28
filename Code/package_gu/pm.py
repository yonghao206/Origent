#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:42:50 2019

This is a package for evaluating different models and compare the performances for Python.

PACKAGE CONTENTS:
    function1: descriptive statistics
    function2: histogram of a column
    function3: boxplot of predicted value for every model
    function4: plot predicted VS observed; plot residual VS observed
    function5: MPE for filtered feature
    function6 and 7: compare the perform of models with filtered feature
    function 8: get the confidence interval 
@author: eileen
"""




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from numpy.polynomial.polynomial import polyfit
    
# function 1
def binned_prediction_summary(data):
    return data.describe().iloc[:,1:].round(3)
 
    
# function2 histogram for the column
def histogram(df, label):
    #df = pd.read_csv('/Users/eileen/UR-Courses/Capstone/prediction.csv')
    a = df[label].hist(grid = False).get_figure()
#    plt.figure(figsize=(18,18))
    plt.title('Histogram of %s Value' % label)
    plt.xlabel('ALSFRS')
    plt.ylabel('Count')
    plt.show()
    # a.savefig('/Users/eileen/UR-Courses/Capstone/python/1.jpg')
    
# function 3
def draw_box(df, category, lower, upper):
    df_new=df.loc[(df[category]>=lower)&(df[category]<=upper)]
    df_two=df_new.copy()

    cols = [i for i in df.columns if 'mod' in i]
    box=[]
    for i in range(len(cols)):
        df_two['residual for '+cols[i]] = df_new['true']-df_new[cols[i]]
        box.append(df_two.boxplot(column='residual for '+cols[i], by=category ,figsize=(12, 6)))

    for i in box:
        plt.show()


# function 5
def MPE_histogram(df, category, lower, upper, model):
    boostrap = 10000
    plt.figure(figsize=(18,18))
    plt.suptitle('MPE for %s' %category, fontsize=16)
    count = 0
    for j in range(lower, upper+1):
        count = count + 1
        MPE = []
        subset = df[df[category] == j]
        for i in range(boostrap):
            sample_M = subset.sample(frac=0.3, replace=True, random_state=i)
            MPE.append(sum(sample_M['true']-sample_M[model])/len(sample_M))
        plot = plt.subplot(4,3, count)
        plot.hist(MPE)
        plot.axvline(0, color = 'r')
        plot.set_title('%s' %j)
    plt.show()

# function 4
from sklearn.metrics import r2_score
import matplotlib.patches as patches

def prediction_summary(df, filter_feature=None, lb=0, ub=10^10):
    if filter_feature is not None:
        df = df[(df[filter_feature]>=lb)&(df[filter_feature]<=ub)]
    

def draw_scatter(df, filter_feature=None, lb=0, ub=10^10):
    if filter_feature is not None:
        df = df[(df[filter_feature]>=lb)&(df[filter_feature]<=ub)]  
    plt.figure(figsize=(14,40))
    cols = [i for i in df.columns if 'mod' in i]
    count=0
    
    for i in range(len(cols)):
        count = count + 1
        x = df['true']
        y = df[cols[i]]
        b, m = np.polyfit(y, x, 1)
        RMSE=np.sqrt(((y- x) ** 2).mean())
        R_square=r2_score(x,y)
   	 
        plot = plt.subplot(6, 2, count)
        plot.scatter(y, x,color='black',s=10)
        plot.plot(y, m + b * y,'-', color='red')
        plot.set_xlabel('model '+str(i+1)+' Predicted')
        plot.set_ylabel('Observed')
        plot.set_title("model "+str(i+1)+" Observed vs Predicted")
        t1=" — RMSE = %.3f"%RMSE
        t2=" — R^2 = %.3f"%R_square
        t3=' — Intercept = %.3f'%m
        t4=' — Slope = %.3f'%b
   	 
        plt.text(0.6, 47, t2, size=8,ha="left", va="bottom",wrap=True)
        plt.text(0.6, 45, t3, size=8,ha="left", va="bottom",wrap=True)
        plt.text(0.6, 43, t1, size=8,ha="left", va="bottom",wrap=True)
        plt.text(0.6, 41, t4, size=8,ha="left", va="bottom",wrap=True)
   	 
        plt.xlim(-2,50)
        plt.ylim(-2,50)
   	 
        rect=patches.Rectangle((-2, 41),16,19,linewidth=1,edgecolor='black',facecolor='none')
        plot.add_patch(rect)
 	 
        count = count + 1
        x = df['true']
        y = df['true'] - df[cols[i]]
        plot = plt.subplot(6, 2, count)
        plt.hlines(0, 0, 51,color='red')
        plot.scatter(x, y,color='black',s=10)
        plot.set_xlabel('observed')
        plot.set_ylabel('model ' + str(i+1) + ' residuals')
        plot.set_title("model "+str(i+1)+" observed vs residuals")
    plt.show()
# =============================================================================
# def draw_scatter(df):
#     plt.figure(figsize=(18,40))
#     cols = [i for i in df.columns if 'mod' in i]
#     count=0
#     for i in range(len(cols)):
#         count = count + 1
#         x = df['true']
#         y = df[cols[i]]
#         b, m = polyfit(x, y, 1)
#         plot = plt.subplot(6, 2, count)
#         plot.scatter(x, y)
#         plot.plot(x, b + m * x, '-', color='red')
#         plot.set_xlabel('observed')
#         plot.set_ylabel('model '+str(i+1)+' predicted')
# 	
#         RMSE=np.sqrt(((y- x) ** 2).mean())
#         R_square=r2_score(x,y)
# 
#         t1="RMSE: %.3f"%RMSE
#         t2="R_square: %.3f"%R_square
#         t4='Slope: %.3f'%m
#         t3='Intercept: %.3f'%b
#         
# 
#         plt.text(0.6, 47, t1, size=8,ha="left", va="bottom",wrap=True)
#         plt.text(0.6, 45, t2, size=8,ha="left", va="bottom",wrap=True)
#         plt.text(0.6, 43, t4, size=8,ha="left", va="bottom",wrap=True)
#         plt.text(0.6, 41, t3, size=8,ha="left", va="bottom",wrap=True)
#    	 
#         rect=patches.Rectangle((-0.5, 41),10,9,linewidth=1,edgecolor='black',facecolor='none')
#         plot.add_patch(rect)
#         plot.set_title("model "+str(i+1)+" observed vs predicted")
#         
#         count = count + 1
#         x = df['true']
#         y = df['true'] - df[cols[i]]
#         b, m = polyfit(x, y, 1)
#         plot = plt.subplot(6, 2, count)
#         plot.scatter(x, y)
#         plot.plot(x, b + m * x, '-', color='red')
#         plot.set_xlabel('observed')
#         plot.set_ylabel('model ' + str(i+1) + ' residuals')
#         plot.set_title("model "+str(i+1)+" observed vs residuals")
# =============================================================================
# =============================================================================
# 
# def draw_scatter(df):
#     plt.figure(figsize=(18,40))
#     cols = [i for i in df.columns if 'mod' in i]
#     count=0
#     for i in range(len(cols)):
#         count = count + 1
#         x = df['true']
#         y = df[cols[i]]
#         b, m = polyfit(x, y, 1)
#         plot = plt.subplot(6, 2, count)
#         plot.scatter(x, y)
#         plot.plot(x, b + m * x, '-', color='red')
#         plot.set_xlabel('observed')
#         plot.set_ylabel('model '+str(i+1)+' predicted')
#         plot.set_title("model "+str(i+1)+" observed vs predicted")
#         
#         count = count + 1
#         x = df['true']
#         y = df['true'] - df[cols[i]]
#         b, m = polyfit(x, y, 1)
#         plot = plt.subplot(6, 2, count)
#         plot.scatter(x, y)
#         plot.plot(x, b + m * x, '-', color='red')
#         plot.set_xlabel('observed')
#         plot.set_ylabel('model ' + str(i+1) + ' residuals')
#         plot.set_title("model "+str(i+1)+" observed vs residuals")
# 
# =============================================================================
    
# function 6 function 7
def prediction_summary(df, filter_feature=None, lb=0, ub=10^10):
    if filter_feature is not None:
        df = df[(df[filter_feature]>=lb)&(df[filter_feature]<=ub)]
    
    cols  = [i for i in df.columns if 'mod' in i]

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy.stats import skew

    row_name = ['R^2', 'RMSE', 'Slope', 'Intercept',  'Skewness']

    label = 'true'
    temp_value =[ [] for i in (cols)]
    for ind, feature in enumerate(cols):
        temp_value[ind].append(r2_score(df[label], df[feature]))
        temp_value[ind].append(np.sqrt(mean_squared_error(df[label], df[feature])))
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(df[feature].values.reshape(-1, 1), df[label])
        temp_value[ind].append(reg.coef_[0])
        temp_value[ind].append(reg.intercept_ )
        temp_value[ind].append(skew(df[feature]))
    temp_table = pd.DataFrame(temp_value).T
    temp_table.columns=cols
    temp_table.index=row_name
    return temp_table.round(3)

#function 8
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def confidence_interval(df, category, lower, upper, model, confidence = 0.95):
    lower_bound = []
    upper_bound = []
    level = []
    bootstrap = 10000
    count = 0
    for j in range(lower, upper+1):
        MPE = []
        count = count + 1
        subset = df[df[category] == j]
        for i in range(bootstrap):
            sample_M = subset.sample(frac=0.3, replace=True, random_state=i)
            MPE.append(sum(sample_M['true']-sample_M[model])/len(sample_M))
        lower_bound.append(mean_confidence_interval(MPE)[0])
        upper_bound.append(mean_confidence_interval(MPE)[1])
        level.append('Month %s' %j)
    col_names =  ['level','lower_bound','upper_bound']
    conf_dat = pd.DataFrame(columns = col_names) 
    conf_dat['level'] = level
    conf_dat['lower_bound'] = lower_bound
    conf_dat['upper_bound'] = upper_bound
    return conf_dat
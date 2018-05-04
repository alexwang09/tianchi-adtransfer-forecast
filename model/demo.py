#!/usr/bin/env python
#-*- coding:utf-8 -*-
# This demo based on IJCAI-2018 CVR prediction

import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np
import time

def prepareData():
    """prepare you dataset here"""
    df = pd.read_csv('../data/offline_dataset.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    return df

def modelscore(y_test, y_pred):
    """set up the evaluation score"""
    return log_loss(y_test, y_pred)

def validation(X,y,features, clf,lossfunction):
    """set up your validation method"""
    totaltest = 0
    for D in [24]:
        T = (X.day != D)
        X_train, X_test = X[T], X[~T]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[T], y[~T]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
        totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
    totaltest /= 1.0
    return totaltest

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}

def main():

    PotentialAdd = ['min_query_time_gap_after', 'hour', 'shop_score_delivery', 'min_query_time_gap_before_user_item', 'shop_id_smooth_query_rate',
    'min_query_time_gap_before', 'shop_score_description', 'item_sales_level', 'shop_query_count', 'user_star_level', 'user_age_level', 'item_sales_query_rate',
    'item_query_count', 'shop_score_service', 'shop_review_positive_rate', 'item_price_level', 'min_query_time_gap_after_user_item']
    '''
    PotentialAdd = []
    '''
    sf = FS.Select(Sequence = True, Random = True, Cross = False, PotentialAdd = PotentialAdd) #select the way you want to process searching
    sf.ImportDF(prepareData(),label = 'is_trade')
    sf.ImportLossFunction(modelscore,direction = 'descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.NonTrainableFeatures = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_id', 'shop_id', 'item_category_0', 'time',
                'context_timestamp', 'item_property_list', 'predict_category_property',
                'item_category_list', 'is_trade', 'day', ]
    sf.InitialFeatures(['item_price_level', 'item_sales_level', 'item_collected_level', 'min_query_time_gap_after', 'min_query_time_gap_before_user_item',
    'min_query_time_gap_after_user_item', 'hour', 'item_category_1', 'shop_score_service', 'user_age_level', 'user_star_level', 'context_page_id', 'min_query_time_gap_before',
    'shop_query_count', 'item_sales_count'])
    #sf.InitialFeatures(['item_price_level','item_sales_level','item_collected_level', 'item_pv_level'])
    sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=8)
    sf.logfile = 'record.log'
    sf.run(validation)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Time used: {}小时'.format((end-start)/3600))

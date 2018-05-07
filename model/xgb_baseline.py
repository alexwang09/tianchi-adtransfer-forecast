# -*- coding: utf-8 -*-

import time
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from bayes_smooth import BayesianSmoothing

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data

def convert_id_feature(trainset, testset):
    tradedset = trainset[trainset.is_trade == 1]
    t1 = trainset.groupby(['item_id']).agg('size').reset_index().rename(columns={0: 'item_id_query_count'})
    t2 = tradedset.groupby(['item_id']).agg('size').reset_index().rename(columns={0: 'item_id_sales_count'})

    t12 = pd.merge(t1, t2, on=['item_id'], how='left')

    t12.item_id_sales_count.fillna(0, inplace=True)
    t12['item_id_smooth_query_rate'] = t12.item_id_sales_count / t12.item_id_query_count

    bs = BayesianSmoothing(1, 1)
    bs.update(t12.item_id_query_count.values,t12.item_id_sales_count.values, 500, 0.0000000001)
    t12['item_id_smooth_query_rate'] = (t12.item_id_sales_count + bs.alpha) / (t12.item_id_query_count + bs.alpha + bs.beta)
    trainset = pd.merge(trainset, t12, on=['item_id'], how='left')
    testset = pd.merge(testset, t12, on=['item_id'], how='left')

    t1 = trainset.groupby(['user_id']).agg('size').reset_index().rename(columns={0: 'user_id_query_count'})
    t2 = tradedset.groupby(['user_id']).agg('size').reset_index().rename(columns={0: 'user_id_sales_count'})

    t12 = pd.merge(t1, t2, on=['user_id'], how='left')

    t12.user_id_sales_count.fillna(0, inplace=True)
    t12['user_id_smooth_query_rate'] = t12.user_id_sales_count / t12.user_id_query_count

    bs = BayesianSmoothing(1, 1)
    bs.update(t12.user_id_query_count.values,t12.user_id_sales_count.values, 500, 0.0000000001)
    t12['user_id_smooth_query_rate'] = (t12.user_id_sales_count + bs.alpha) / (t12.user_id_query_count + bs.alpha + bs.beta)
    print pd.value_counts(t12['user_id_smooth_query_rate'])
    trainset = pd.merge(trainset, t12, on=['user_id'], how='left')
    testset = pd.merge(testset, t12, on=['user_id'], how='left')
    #wait for validation
    #t5 = dataset.groupby(['shop_id']).agg('size').reset_index().rename(columns={0: 'total_shop_query_count'})
    #trainset = pd.merge(trainset, t5, on=['shop_id'], how='left')
    #testset = pd.merge(testset, t5, on=['shop_id'], how='left')

    return trainset, testset

if __name__ == "__main__":
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    #去掉了instance_id,context_id,item_category_list,item_property_list,predict_category_property,user_id,context_timestamp
    features = ['item_id_smooth_query_rate', 'user_id_smooth_query_rate']
    target = ['is_trade']
    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集

    elif online == True:
        train = data.copy()
        test = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt', sep=' ')

    train, test = convert_id_feature(train, test)

    trainset = xgb.DMatrix(train[features],label=train[target])
    testset = xgb.DMatrix(test[features],label=test[target])
    te = xgb.DMatrix(test[features])
    params={'booster':'gbtree',
    	    'objective': 'binary:logistic',
     	    'eval_metric':'logloss',
    	    #'gamma':0.1,
    	    'min_child_weight':1,
    	    'max_depth':5,
    	    #'lambda':10,

    	    'eta': 0.05,
    	    'tree_method':'exact',
    	    'seed':0,
    	    'nthread':12
    	    }
    if online == False:
        watchlist = [(trainset,'train'),(testset,'val')]
        model = xgb.train(params,trainset,num_boost_round=3000, evals=watchlist, early_stopping_rounds=100)
        print model.best_ntree_limit
        print model.get_score()
        count = 0
        for k, v in model.get_score(importance_type='gain').iteritems():
            count += v
        print count
        test['lgb_predict'] = model.predict(te, ntree_limit=model.best_ntree_limit)
        #test.lgb_predict = MinMaxScaler((np.exp(-10), 1-np.exp(-10))).fit_transform(np.array(test.lgb_predict).reshape(-1, 1))

        print(log_loss(test[target], test['lgb_predict']))
    else:
        model = xgb.train(params,trainset,num_boost_round=138)

        test['predicted_score'] = model.predict(te)
        test[['instance_id', 'predicted_score']].to_csv(r'../submit/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                                                        index=False,sep=' ')#保存在线提交结果
        #test.lgb_predict = MinMaxScaler((np.exp(-10), 1-np.exp(-10))).fit_transform(np.array(test.lgb_predict).reshape(-1, 1))

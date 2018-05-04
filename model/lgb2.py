# -*- coding: utf-8 -*-

import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import datetime, copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from bayes_smooth import BayesianSmoothing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def map_hour(x):
    if x>=21:
        return 0
    if x>=7 and x<=12:
        return 1
    elif x>13 and x<=20:
        return 2
    return 3

def convert_timestamp(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    #data['hour_seg'] = data.hour.map(map_hour)
    return data

def convert_category_feature(data):
    for i in range(3):
        data['item_category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
            )

    #data['item_property_count'] = data['item_property_list'].apply(lambda x:len(x.split(";")) )
    return data

def get_min_query_time_gap_before(s):
    time_list,current_time = s.split('-')
    times = time_list.split(',')
    gaps = []
    for t in times:
        this_gap = int(current_time) - int(t)
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)

def get_min_query_time_gap_after(s):
    time_list,current_time = s.split('-')
    times = time_list.split(',')
    gaps = []
    for t in times:
        this_gap = int(t) - int(current_time)
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)

def is_price_level_changed_after_last_query(s):
    price_time_list, current_price_time = s.split('-')
    times = [price_time.split(':')[0] for price_time in price_time_list.split(',')]
    price = [price_time.split(':')[1] for price_time in price_time_list.split(',')]
    current_time = current_price_time.split(':')[0]
    current_price = current_price_time.split(':')[1]
    gaps = []
    min_gap = float('inf')
    index = -1
    for i, t in enumerate(times):
        this_gap = int(current_time) - int(t)
        if this_gap>0 and this_gap < min_gap:
            min_gap = this_gap
            index = i
    if index == -1:
        return -1
    else:
        if price[index] == current_price:    return 0
        return 1

def has_bought_before(s):
    if s == 0:  return 0
    time_list, current_time = s.split('-')
    buy_item_min_time = min(time_list.split(','))
    return 1 if buy_item_min_time < current_time else 0

def check_order_of_this_query(s):
    before_gap, after_gap = s.split(',')
    before_gap = int(before_gap)
    after_gap = int(after_gap)
    if after_gap == -1:    return 0
    return 1

def extract_overall_features(train, test):
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train['instance_user_id'] = train.instance_id.astype('str') + '-' + train.user_id.astype('str')
    test['instance_user_id'] = test.instance_id.astype('str') + '-' + test.user_id.astype('str')
    train_id = train.instance_user_id.values.copy()
    test_id = test.instance_user_id.values.copy()
    data = pd.concat([train, test])

    t1 = data[['user_id', 'context_timestamp']]
    t1.context_timestamp = t1.context_timestamp.astype('str')
    t1 = t1.groupby(['user_id'])['context_timestamp'].agg(lambda x: ','.join(x)).reset_index()
    t1.rename(columns={'context_timestamp': 'context_timestamp_list'}, inplace=True)
    #t1['max_context_timestamp_query'] = t1.context_timestamp_list.apply(lambda s: max([d for d in s.split(',')]))
    #t1['min_context_timestamp_query'] = t1.context_timestamp_list.apply(lambda s: min([d for d in s.split(',')]))
    data = pd.merge(data, t1, on=['user_id'], how='left')
    #data['time_after_first_query'] = data.context_timestamp - data.min_context_timestamp_query.map(int)
    #data['time_before_last_query'] = -data.context_timestamp + data.max_context_timestamp_query.map(int)
    data.context_timestamp_list = data.context_timestamp_list + '-' + data.context_timestamp.astype('str')

    data['min_query_time_gap_before'] = data.context_timestamp_list.map(get_min_query_time_gap_before)
    data['min_query_time_gap_after'] = data.context_timestamp_list.map(get_min_query_time_gap_after)
    data.drop(['context_timestamp_list'], axis=1, inplace=True)
    data['min_query_minute_gap_before'] = data.min_query_time_gap_before.map(lambda x: int(x)/60 if x > 0 else -1)


    #data['order_of_this_query'] = data.min_query_time_gap_before.astype('str') + ',' + data.min_query_time_gap_after.astype('str')
    #data.order_of_this_query = data.order_of_this_query.map(check_order_of_this_query)

    t2 = data[['user_id', 'item_id', 'context_timestamp']]
    t2.context_timestamp = t2.context_timestamp.astype('str')
    t2 = t2.groupby(['user_id', 'item_id'])['context_timestamp'].agg(lambda x: ','.join(x)).reset_index()
    t2.rename(columns={'context_timestamp': 'context_timestamp_list_user_item'}, inplace=True)
    data = pd.merge(data, t2, on=['user_id', 'item_id'], how='left')
    data.context_timestamp_list_user_item = data.context_timestamp_list_user_item + '-' + data.context_timestamp.astype('str')
    data['min_query_time_gap_before_user_item'] = data.context_timestamp_list_user_item.map(get_min_query_time_gap_before)
    data['min_query_time_gap_after_user_item'] = data.context_timestamp_list_user_item.map(get_min_query_time_gap_after)
    data.drop(['context_timestamp_list_user_item'], axis=1, inplace=True)


    train = data[data.instance_user_id.isin(train_id)]
    test = data[data.instance_user_id.isin(test_id)]

    del train['instance_user_id']
    del test['instance_user_id']

    #convert item_property_list

    return train, test

def onehot_encode(train, testa):
    enc = OneHotEncoder()
    lb = LabelEncoder()
    feat_set = ['item_category_1', 'item_category_2', 'item_city_id']
    for i,feat in enumerate(feat_set):
        tmp = lb.fit_transform((list(train[feat])+list(testa[feat])))
        tmp1 = tmp[:len(train.index)]
        tmp2 = tmp[len(train.index):]
        train[feat] = pd.Series(tmp1)
        testa[feat] = pd.Series(tmp2)
    return train, testa

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = log_loss(label,pred)
    return ('logloss',score,False)

def extract_feature_from_former_days(feature, dataset, i):
        #################item related feature#####################
        """
        1.item related:
            item_query_count
            item_sales_count
            item_sales_query_rate
        """
        #dataset['item_price/item_sales'] = ((dataset['item_price_level']+1.0) / (dataset['item_sales_level']+1.0)).map(lambda x: -1 if x == float('inf') else x)
        #dataset['item_price*item_sales'] = (dataset['item_price_level'] * dataset['item_sales_level']).map(lambda x: -1 if x<0 else x)
        #dataset['item_price+item_sales'] = dataset['item_price_level'] + dataset['item_sales_level']

        t1 = feature.groupby(['item_id']).agg('size').reset_index().rename(columns={0: 'item_query_count'})
        t2 = feature[feature.is_trade == 1]
        t2 = t2.groupby(['item_id']).agg('size').reset_index().rename(columns={0: 'item_sales_count'})

        dataset = pd.merge(dataset, t1, on=['item_id'], how='left')
        dataset = pd.merge(dataset, t2, on=['item_id'], how='left')

        dataset.item_sales_count = dataset.item_sales_count.fillna(0)
        dataset['item_sales_query_rate'] = dataset.item_sales_count / dataset.item_query_count
        dataset.item_query_count = dataset.item_query_count.fillna(0)

        #dataset.item_brand_sales_count = dataset.item_brand_sales_count.fillna(0)
        #dataset['item_brand_sales_query_rate'] = dataset.item_brand_sales_count / dataset.item_brand_query_count
        #dataset.item_brand_query_count = dataset.item_brand_query_count.fillna(0)

        #################item related feature#####################
        """
        2.user related:
            user_query_count
            user_query_buy_rate
        """

        t1 = feature.groupby(['user_id']).agg('size').reset_index().rename(columns={0: 'user_query_count'})

        t2 = feature[feature.is_trade == 1]
        t2 = t2.groupby(['user_id']).agg('size').reset_index().rename(columns={0: 'user_buy_count'})
        t12 = pd.merge(t1, t2, on=['user_id'], how='left')
        t12.user_buy_count = t12.user_buy_count.fillna(0)
        #t2['user_buy_query_rate'] = t2.user_buy_count / t2.user_query_count

        #bs = BayesianSmoothing(1, 1)
        #bs.update(t12.user_query_count.values, t12.user_buy_count.values, 1000, 0.0000000001)
        t12['user_id_smooth_query_rate'] = (t12.user_buy_count) / (t12.user_query_count)
        dataset = pd.merge(dataset, t12, on=['user_id'], how='left')

        dataset.user_query_count.fillna(0, inplace=True)
        del dataset['user_buy_count']

        #################item related feature#####################
        """
        3.context related:

        """

        #################item related feature#####################
        """
        4.shop related:
            shop_query_count
            shop_sales_count
            shop_sales_query_rate
        """
        t1 = feature.groupby(['shop_id']).agg('size').reset_index().rename(columns={0: 'shop_query_count'})

        t2 = feature[feature.is_trade == 1]
        t2 = t2.groupby(['shop_id']).agg('size').reset_index().rename(columns={0: 'shop_sales_count'})

        #dataset = pd.merge(dataset, t1, on=['shop_id'], how='left')
        #dataset = pd.merge(dataset, t2, on=['shop_id'], how='left')
        #dataset.shop_sales_count = dataset.shop_sales_count.fillna(0)
        #dataset['shop_sales_query_rate'] = dataset.shop_sales_count / dataset.shop_query_count
        #dataset.shop_query_count = dataset.shop_query_count.fillna(0)

        t12 = pd.merge(t1, t2, on=['shop_id'], how='left')
        t12.shop_sales_count = t12.shop_sales_count.fillna(0)

        #bs = BayesianSmoothing(1, 1)
        #bs.update(t12.shop_query_count.values, t12.shop_sales_count.values, 1000, 0.0000000001)
        t12['shop_id_smooth_query_rate'] = (t12.shop_sales_count) / (t12.shop_query_count)
        dataset = pd.merge(dataset, t12, on=['shop_id'], how='left')
        dataset.shop_query_count.fillna(0, inplace=True)
        dataset.shop_sales_count.fillna(0, inplace=True)
        #del dataset['shop_sales_count']

        #################user_item related feature#####################
        """
        5.user_item related:
            user_query_same_item_count
            user_buy_same_item_rate

        """

        t1 = feature.groupby(['user_id', 'item_id']).agg('size').reset_index().rename(columns={0: 'user_query_same_item_id_count'})
        t2 = feature[feature.is_trade == 1]
        t2 = t2.groupby(['user_id', 'item_id']).agg('size').reset_index().rename(columns={0: 'user_buy_same_item_id_count'})

        dataset = pd.merge(dataset, t1, on=['user_id', 'item_id'], how='left')
        dataset = pd.merge(dataset, t2, on=['user_id', 'item_id'], how='left')
        dataset['user_buy_same_item_id_count'].fillna(0, inplace=True)
        dataset['user_buy_same_item_id_rate'] = dataset.user_buy_same_item_id_count / dataset.user_query_same_item_id_count
        dataset['user_query_same_item_id_count'].fillna(0, inplace=True)
        del dataset['user_buy_same_item_id_count']

        """
        6.user_context related:
        """

        """
        7.user_shop related:
        """

        return dataset


def extract_feature_from_this_day(df):
    ###################other feature##########################

    """
    8. other features:
        this_day_user_query_count
        this_day_hour_user_query_count
        #this_day_user_query_same_item_count
        this_day_user_query_same_shop_count
        the_hourgap_user_query_same_item_lastone
        the_hourgap_user_query_same_item_nextone
        min_hour_gap_after
        min_hour_gap_before

    """

    df['is_user_gender_missing'] = df.user_gender_id.map(lambda x: 1 if x == -1 else 0)

    t1 = df.groupby(['user_id']).agg('size').reset_index().rename(columns={0: 'this_day_user_query_count'})
    t2 = df.groupby(['user_id', 'hour']).agg('size').reset_index().rename(columns={0: 'this_day_hour_user_query_count'})
    #t4 = df.groupby(['user_id', 'item_id']).agg('size').reset_index().rename(columns={0: 'this_day_user_query_same_item_count'})
    t3 = df.groupby(['user_id', 'shop_id']).agg('size').reset_index().rename(columns={0: 'this_day_user_query_same_shop_count'})
    #t4 = df.groupby(['shop_id']).agg('size').reset_index().rename(columns={0: 'this_day_shop_query_count'})

    df = pd.merge(df, t1, on=['user_id'], how='left')
    df = pd.merge(df, t2, on=['user_id', 'hour'], how='left')
    #df = pd.merge(df, t4, on=['user_id', 'item_id'], how='left')
    df = pd.merge(df, t3, on=['user_id', 'shop_id'], how='left')
    #df = pd.merge(df, t4, on=['shop_id'], how='left')

    return df


if __name__ == "__main__":
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=' ')
    testa = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    testb = pd.read_csv('../data/round1_ijcai_18_test_b_20180418.txt', sep=' ')

    testa_instance_id = testa[['instance_id']]
    testb_instance_id = testb[['instance_id']]
    testa = pd.concat([testa, testb])
    testa = testa.reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)
    testa = data.replace(-1, np.nan)
    testa = testa.replace(-1,np.nan)
    data = convert_timestamp(data)
    testa = convert_timestamp(testa)
    data = convert_category_feature(data)
    testa = convert_category_feature(testa)

    #data = extract_feature(data)
    #testa = extract_feature(testa)
    data, testa = onehot_encode(data, testa)

    #data, testa = extract_overall_features(data, testa)

    #split the dataset
    feature1 = data[(data.day >= 18) & (data.day <= 18)]
    dataset1 = data[data.day == 19]
    feature2 = data[(data.day >= 19) & (data.day <= 19)]
    dataset2 = data[data.day == 20]
    feature3 = data[(data.day >= 20) & (data.day <= 20)]
    dataset3 = data[data.day == 21]
    feature4 = data[(data.day >= 21) & (data.day <= 21)]
    dataset4 = data[data.day == 22]
    feature5 = data[(data.day >= 22) & (data.day <= 22)]
    dataset5 = data[data.day == 23]
    feature6 = data[(data.day >= 23) & (data.day <= 23)]
    dataset6 = data[data.day == 24]
    feature7 = data[(data.day >= 24) & (data.day <= 24)]
    '''
    dataset3 only has the 30% data of the day 25
    '''
    dataset7 = testa

    dataset7 = extract_feature_from_former_days(feature7, dataset7, 0)
    dataset6 = extract_feature_from_former_days(feature6, dataset6, 1)
    dataset5 = extract_feature_from_former_days(feature5, dataset5, 2)
    dataset4 = extract_feature_from_former_days(feature4, dataset4, 3)
    dataset3 = extract_feature_from_former_days(feature3, dataset3, 4)
    dataset2 = extract_feature_from_former_days(feature2, dataset2, 5)
    dataset1 = extract_feature_from_former_days(feature1, dataset1, 6)

    dataset7 = extract_feature_from_this_day(dataset7)
    dataset6 = extract_feature_from_this_day(dataset6)
    dataset5 = extract_feature_from_this_day(dataset5)
    dataset4 = extract_feature_from_this_day(dataset4)
    dataset3 = extract_feature_from_this_day(dataset3)
    dataset2 = extract_feature_from_this_day(dataset2)
    dataset1 = extract_feature_from_this_day(dataset1)


    #######################modeling and training###############################

    if online == False:
        train = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])
        test = dataset6

    elif online == True:
        train = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6])
        test = dataset7
    #train = train.reset_index(drop=True)
    train, test = extract_overall_features(train, test)

    if online == True:
        test = pd.merge(testb_instance_id, test, on=['instance_id'], how='left')

    drop_features = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_id', 'shop_id', 'item_category_0', 'time',
                'context_timestamp', 'item_property_list', 'predict_category_property',
                'item_category_list', 'is_trade', 'day', ]

    #add_drop =  (['min_query_time_gap_before', 'is_user_gender_missing', 'user_query_same_item_id_count',
    #'item_price_level', 'user_query_count', 'user_buy_same_item_id_rate', 'shop_sales_count', 'shop_score_service', 'shop_review_num_level', 'item_sales_count', 'shop_review_positive_rate', 'shop_star_level', 'this_day_hour_user_query_count', 'item_category_2', 'user_occupation_id']
    features = list(set(test.columns) - set(drop_features))
    target = 'is_trade'

    params = {
        'learning_rate':0.05,
        'objective':'binary',
        'num_leaves':64,
        'max_depth':7,
        #'scale_pos_weight':50,
        #'poisson_max_delta_step':1
    }
    categorical_feature = ['user_gender_id', 'item_category_1', 'item_category_2', 'user_occupation_id',
                         ]
    if online == False:
        #x_train,x_test,y_train,y_test=train_test_split(train[],label_all,test_size=0.2,random_state=42)
        #for i, features in enumerate(feature_list):
        d_train = lgb.Dataset(train[features], label=train[target], categorical_feature=categorical_feature)
        d_test = lgb.Dataset(test[features], test[target], categorical_feature=categorical_feature)

        clf = lgb.train(params, d_train,
                        num_boost_round=3000,
                        valid_sets=[d_train,d_test],
                        verbose_eval=200,
                        feval=evalerror,
                        early_stopping_rounds=200)

        #num_iteration这里加不加都一样
        test['lgb_predict'] = clf.predict(test[features],num_iteration=clf.best_iteration)
        #test['lgb_predict'] = (test['lgb_predict0'] + test['lgb_predict1'])/2
        print log_loss(test[target], test['lgb_predict'])

    else:
        d_train = lgb.Dataset(train[features], label=train[target], categorical_feature=categorical_feature)
        clf = lgb.train(params, d_train,
                        num_boost_round=194,
                        valid_sets=d_train,
                        verbose_eval=30,
                        feval=evalerror,
                        )
        print sorted(zip(clf.feature_importance(), features))[::-1]
        #num_iteration这里加不加都一样
        test['predicted_score'] = clf.predict(test[features],num_iteration=clf.best_iteration)
        test[['instance_id', 'predicted_score']].to_csv(r'../submit/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                                                        index=False,sep=' ')#保存在线提交结果

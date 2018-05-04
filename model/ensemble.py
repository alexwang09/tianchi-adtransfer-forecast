# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


sub1 = pd.read_csv('../submit/sub20180422_094132.csv', sep=' ')
sub2 = pd.read_csv('../submit/sub20180420_235931.csv', sep=' ')
sub = pd.merge(sub1, sub2, on=['instance_id'], how='left')
sub['predicted_score'] = (sub.predicted_score1 + sub.predicted_score2)/2
sub[['instance_id', 'predicted_score']].to_csv(r'../submit/sub.csv',index=False,sep=' ')
'''
data = pd.read_csv('../data/dataset2.csv', sep=' ')
#testb = pd.read_csv('../data/round1_ijcai_18_test_b_20180418.txt', sep=' ')
data = data[data.day == 24]
data = data[['instance_id', 'user_id', 'is_trade']]
data['instance_user_id'] = data.instance_id.astype('str') + '-' + data.user_id.astype('str')
valid1['instance_user_id'] = valid1.instance_id.astype('str') + '-' + valid1.user_id.astype('str')
del valid1['instance_id']
del valid1['user_id']
valid2['instance_user_id'] = valid2.instance_id.astype('str') + '-' + valid2.user_id.astype('str')
del valid2['instance_id']
del valid2['user_id']
data = pd.merge(data, valid1, on=['instance_id'], how='left')
data = pd.merge(data, valid2, on=['instance_id'], how='left')
'''

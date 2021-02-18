# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 00:10:16 2021

@author: Sergei
"""
import pandas as pd
import numpy as np

def pairs_quality(df_true, df_pred, y=2):
    """
    Функция считает precision и recall для пар новостей
    :param df_true: pd.DataFrame, у которого индекс -- пара новостей и стобец 'res' с настоящей меткой этой пары
    :param df_pred:pd.DataFrame, у которого индекс -- пара новостей и стобец 'res' с предсказанной меткой этой пары
    :param y: уровень измерения качества (0, 1, 2)
    :return: пара (precision, recall)
    """
    shared_pairs = df_true.index.intersection(df_pred.index)

    inds_1 = (df_pred.loc[shared_pairs]['res'] >= y)
    inds_2 = (df_true.loc[shared_pairs]['res'] >= y)

    num = (inds_1 & inds_2).sum()

    precision = num / inds_1.sum()
    recall = num / inds_2.sum()

    return precision, recall

def chain_quality(df_true, chains_list, y=2):
    cluster_dict = {}
    for i, chain in enumerate(chains_list):
        for ind in chain:
            cluster_dict[ind] = i
    
    num = 0
    denum_prec = 0
    denum_rec = 0
    for pair in df_true.index:
            clust_1 = cluster_dict[pair[0]]
            clust_2 = cluster_dict[pair[1]]
            res = df_true.loc[pair]['res']
            if (clust_1 == clust_2) and (res >= y):
                num += 1
            if clust_1 == clust_2:
                denum_prec += 1
            if res >= y:
                denum_rec += 1

    
    precision = num / denum_prec
    recall = num / denum_rec
    return precision, recall
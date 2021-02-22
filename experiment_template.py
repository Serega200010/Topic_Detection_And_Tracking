# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:11:29 2021

@author: Sergei
"""

import numpy as np

from sklearn.preprocessing import LabelEncoder
from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import re
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from distances import*
from ewm import *
from loss import *
import sklearn
def preprocessor(text: str) -> str:
    text = re.sub(r'[!\"#$%&\'()*\+,-./:;<=>?@\[\\\]^_`{|}~]+', ' ', text.lower())
    text = text.replace('ё', 'е')
    return text


def stemm_tokenizer(text: str, stemmer, tknzr) -> str:
    cleaned_text = preprocessor(text)
    stemmed_text = ' '.join([stemmer.stem(w) for w in tknzr.tokenize(cleaned_text) if w.isalpha()])
    return stemmed_text

class Experiment:
    """
    Класс-шаблон для создания своих экспериментов.
    Для создания своего эксперимента, нужно унаследоваться от этого класса и
    переопределить функции dist_func и vec_func.
    Пример использования смотрите в notebooks/experiment_example.ipynb
    """
    def __init__(self, data, thresh=None):
        self.data = data
        
        self.le = LabelEncoder()
        self.chains_list = []
        self.X = None
        self.dist_matr = None
        self.thresh = thresh

    def dist_func(self, dist_vec: np.array, chain_vecs: np.ndarray) -> float:
        """
        Функция расстояния от очередной новости до цепочки
        :param dist_vec: Вектор расстояний от очередной новости до новостей в цепочке 
        :return: Расстояние от новости до цепочки
        """
        pass

    def vec_func(self, ind_list: list) -> np.ndarray:
        """
        Функция векторизации новостей с данными индексами
        :param ind_list: Список индексов из self.data новостей для векторизации
        :return: Матрица векторов соответствующих новостей
        """
        pass
    
    def make_dist_matr(self, X):
        """
        Функция создания матрицы попарных расстояний векторов X
        :param X: Матрица размера (N, D), где в каждой строчке содержится вектор соответствующей новости
        :return: Матрица размера (N, N), где (i, j) элемент содержит расстояние от X[i] до X[j]
        """
        pass

    def _create_new_chain(self, new_i):
        self.chains_list.append([new_i])

    def _add_new_to_chain(self, new_i, chain_i):
        self.chains_list[chain_i].append(new_i)

    def find_chain(self, new_i, new_ind):
        dist_arr = np.zeros(len(self.chains_list))
        for chain_i, chain in enumerate(self.chains_list):
            dist = self.dist_func(self.dist_matr[new_i], self.chains_list[chain_i])
            dist_arr[chain_i] = dist

        if dist_arr.size == 0:
            self._create_new_chain(new_i)
            return

        best_chain_i = dist_arr.argmin()
        best_dist = dist_arr[best_chain_i]

        if best_dist > self.thresh:
            self._create_new_chain(new_i)
            return

        self._add_new_to_chain(new_i, best_chain_i)
    
    def compute_distances(self):
        self.X = self.vec_func(self.data.index)
        self.dist_matr = self.make_dist_matr(self.X)
        
    def create_chains(self, thresh=None) -> list:
        """
        Функция создания списка цепочек из данных из self.data
        :return: слисок всех цепочек
        """
        self.chains_list = []
        if thresh is not None:
            self.thresh = thresh
        
        if self.X is None or self.dist_matr is None:
            self.compute_distances()
        
        self.le.fit(self.data.index)
        for i, ind in enumerate(self.data.index):
            self.find_chain(i, ind)
            
        self.chains_list = [self.le.inverse_transform(chain) for chain in self.chains_list]
        return self.chains_list
    
    




    

"""
Спецификаторы:
- Cos_Mean - Усреднение по всей цепочке функции косинусного расстояния до нового ообъекта
- Cos_WMean - Экспоненциально взвешенное усреднение по всей цепочке функции косинусного расстояния до нового ообъекта
- Cos_Min - Минимум косинусного расстояния по всем векторам из цепочки
- Cos_Last - Косинусное расстояние до последнего вектора из цепочки

- Euc_Mean - Усреднение по всей цепочке функции евклидового расстояния до нового ообъекта
- Euc_WMean - Экспоненциально взвешенное усреднение по всей цепочке функции евклидового расстояния до нового ообъекта
- Euc_Min - Минимум евклидового расстояния по всем векторам из цепочки
- Euc_Last - Евклидово расстояние до последнего вектора из цепочки

- Man_Mean - Усреднение по всей цепочке функции манхеттенского расстояния до нового ообъекта
- Man_WMean - Экспоненциально взвешенное усреднение по всей цепочке функции манхеттенского расстояния до нового ообъекта
- Man_Min - Минимум манхеттенского расстояния по всем векторам из цепочки
- Man_Last - Манхетенское расстояние до последнего вектора из цепочки

- KL_Mean - Усреднение по всей цепочке функции дивергенции KL до нового ообъекта
- KL_WMean - Экспоненциально взвешенное усреднение по всей цепочке дивергенции KL до нового ообъекта
- KL_Min - Минимум дивергенции KL по всем векторам из цепочки
- KL_Last - Дивергенция KL расстояние до последнего вектора из цепочки
"""


    
    
class MyExperiment(Experiment):
    def __init__(self, data, tfidf, spec = "Cos_Mean", treshold = 1):
        super().__init__(data, treshold) # задаем порог
        self.tfidf = tfidf
        self.spec = spec
        self.treshold = treshold
        self.Spec_dict = {'Cos_Mean' : Cosine_measure_to_chain_av,
                          'Cos_WMean': Cosine_measure_to_chain_w_av,
                          'Cos_Min'  : Cosine_measure_to_chain_asMin,
                          'Cos_Last' : Cosine_measure_to_chain_last,
                         
                          'Euc_Mean' : Euclidian_measure_to_chain_av,
                          'Euc_WMean': Euclidian_measure_to_chain_w_av,
                          'Euc_Min'  : Euclidian_measure_to_chain_asMin,
                          'Euc_Last' : Euclidian_measure_to_chain_last,
                          
                          'Man_Mean' : Manhattan_measure_to_chain_av,
                          'Man_WMean': Manhattan_measure_to_chain_w_av,
                          'Man_Min'  : Manhattan_measure_to_chain_asMin,
                          'Man_Last' : Manhattan_measure_to_chain_last,
                         
                          'KL_Mean' : KL_measure_to_chain_av,
                          'KL_WMean': KL_measure_to_chain_w_av,
                          'KL_Min'  : KL_measure_to_chain_asMin,
                          'KL_Last' : KL_measure_to_chain_last}
        if self.spec[0:3] == 'Cos':
            self.func = lambda u,v: cosine(u,v)
        
        if self.spec[0:3] == 'Euc':
            self.func = lambda u,v: np.linalg.norm(u,v)
            
        if self.spec[0:3] == 'Man':
            self.func = lambda u,v: np.sum(np.abs(u - v))      

        if self.spec[0:3] == 'KL':
            self.func = lambda u,v: entropy(u - v)     
        self.result = {}
    
    # определяем свою функцию dist_func
    def dist_func(self, new_vec, chain_vecs):
        return self.Spec_dict[self.spec](chain_vecs,new_vec)
    
    def new_spec(self, spec = 'Cos_Mean'):
        self.spec = spec
    
    # определяем свою функцию vec_func
    def vec_func(self, ind_list):
        text_list = self.data.loc[ind_list].text
        vect_arr = self.tfidf.transform(text_list).toarray()
        return vect_arr
    
    def make_dist_matr(self, X):
        return sklearn.metrics.pairwise_distances(X,X, metric = self.func)
    
    def run(self):
        chain_list = self.create_chains()
        self.result = {'Specification' : self.spec,
                       'Treshold' : self.treshold,
                       'Num of news' : self.data.shape[0],
                       'Num of chains' : len(chain_list),
                       'Max chain length' : max([len(c) for c in chain_list]),
                       'Min chain length' : min([len(c) for c in chain_list]),
                       'Chains' : chain_list}
        print('Succesful run')
    def info(self):
        print(self.result)
    def save_result(self,file):
        f = open(file, 'wb')
        pickle.dump(self.result, f)
        f.close()
    def read_result(self,file,  set_result = False):
        with open(file, 'rb') as f:
            res = pickle.load(f)
            if set_result:
                self.result = res
        return res
    
    def form_filename(self,number = 0):
        return 'exp_'+self.spec+"_"+str(number) + '.pickle'
    
    def evaluate(self, 
                 #test_data_path = '../data/test_data.csv',
                 toloka_data = '../data/toloka_result_pairs.csv'):
        comm_res = pd.read_csv(toloka_data)
        valid_inds = np.union1d(comm_res['new_1'].values, comm_res['new_2'].values)
        comm_res = comm_res.groupby(['new_1', 'new_2']).max()
        
        comm_res_train, comm_res_test, _, _ = train_test_split(comm_res, comm_res, train_size=0.5, random_state=42)
        #data = pd.read_csv('../data/test_data.csv') # выгружаем датасет новостей
        #data = data[~data['text'].isna()]
        
        score_train = chain_quality(comm_res_train, self.result['Chains'])
        score_test = chain_quality(comm_res_test, self.result['Chains'])
        
        res = {'Pr_train' : score_train[0], 'Rec_train' : score_train[0],
               'Pr_test'  : score_train[0], 'Rec_test'  : score_train[0]}
        print("Result of evaluation: \nTrain precision: {}\nTrain Recall:{}\nTest Precision:{}\nTestRecall:{}".format(score_train[0],score_train[1],score_test[0],score_test[1]))
        
        
'''
class evaluator():
    def __init__(self, experiment, 
                 toloka_data = '../data/toloka_result_pairs.csv', 
                 test_data_path = '../data/test_data.csv'):
        self.experiment = experiment
        self.data = pd.read_csv('../data/test_data.csv')
        self.comm_res = pd.read_csv('../data/toloka_result_pairs.csv')
        self.valid_inds = np.union1d(self.comm_res['new_1'].values, self.comm_res['new_2'].values)
        
        self.data = self.data[~self.data['text'].isna()]
        self.comm_res_train, self.comm_res_test, _, _ = train_test_split(comm_res, comm_res, train_size=0.5, random_state=42)
    
'''
    
    
'''
    def evaluate(self,df_true):
        res = self.result['Chains']
        pairs = get_pairs(res)
        #valid_inds = np.union1d(df_true['new_1'].values,df_true['new_2'].values)
        
        true = df_true.groupby(['new_1', 'new_2']).max()
        #pr, rec  =  pairs_quality(true, pairs)
       # pr, rec  =  chain_quality(true, pairs)
        pr, rec  =  chain_quality(true, res)
        
        print('The evaluation of result was succesfully run\nPrecision:',pr,'\nRecall:',rec)
        return pr, rec
        
    def run_eval(self,df_true):
        self.run()
        pr, rec = self.evaluate(df_true)
        return pr,rec'''
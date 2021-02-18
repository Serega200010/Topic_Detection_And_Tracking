
from numba import njit
from distances import *
import numpy as np
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import re
from scipy.spatial.distance import cosine
import pickle
from loss import pairs_quality, chain_quality
import itertools

def get_pairs(chains_list):
    pair_list = []
    for chain in chains_list:
        pair_list += list(itertools.combinations(sorted(chain), 2))
 
    res = pd.Series({pair: 2 for pair in pair_list}).to_frame('res')
    res.index = res.index.rename(['new_1', 'new_2'])
    return res


class Experiment:
    """
    Класс-шаблон для создания своих экспериментов.
    Для создания своего эксперимента, нужно унаследоваться от этого класса и
    переопределить функции dist_func и vec_func.
    Пример использования смотрите в notebooks/experiment_example.ipynb
    """
    def __init__(self, data, thresh):
        self.data = data
        self.chains_list = []
        self.chains_vecs_list = []
        self.thresh = thresh

    def dist_func(self, new_vec: np.array, chain_vecs: np.ndarray) -> float:
        """
        Функция расстояния от очередной новости до цепочки
        :param new_vec: Вектор новости
        :param chain_vecs: Numpy матрица векторов всех новостей в данной цепочке
        :return: Расстояние от новости до цепочки (число)
        """
        pass

    def vec_func(self, ind_list: list) -> np.ndarray:
        """
        Функция векторизации новостей с данными индексами
        :param ind_list: Список индексов из self.data новостей для векторизации
        :return: Матрица векторов соответствующих новостей
        """
        pass


    def _create_new_chain(self, new_ind, new_vec):
        self.chains_list.append([new_ind])
        self.chains_vecs_list.append(new_vec.reshape(1,-1))


    def _add_new_to_chain(self, new_ind, new_vec, chain_ind):
        self.chains_list[chain_ind].append(new_ind)
        self.chains_vecs_list[chain_ind] = np.vstack([self.chains_vecs_list[chain_ind], new_vec])
     
    def find_chain(self, new_ind):
        dist_arr = np.zeros(len(self.chains_list))
        new_vec = self.vec_func([new_ind])[0]
        for i, chain in enumerate(self.chains_list):
            dist = self.dist_func(new_vec, self.chains_vecs_list[i])
            dist_arr[i] = dist

        if dist_arr.size == 0:
            self._create_new_chain(new_ind, new_vec)
            return

        best_chain_ind = dist_arr.argmin()
        best_dist = dist_arr[best_chain_ind]

        if best_dist > self.thresh:
            self._create_new_chain(new_ind, new_vec)
            return

        self._add_new_to_chain(new_ind, new_vec, best_chain_ind)


    def create_chains(self) -> list:
        """
        Функция создания списка цепочек из данных из self.data
        :return: слисок всех цепочек
        """
        self.chains_list = []
        self.chains_vecs_list = []
        for ind in self.data.index:
            self.find_chain(ind)
        return self.chains_list
    

def preprocessor(text: str) -> str:
    text = re.sub(r'[!\"#$%&\'()*\+,-./:;<=>?@\[\\\]^_`{|}~]+', ' ', text.lower())
    text = text.replace('ё', 'е')
    return text


def stemm_tokenizer(text: str, stemmer, tknzr) -> str:
    cleaned_text = preprocessor(text)
    stemmed_text = ' '.join([stemmer.stem(w) for w in tknzr.tokenize(cleaned_text) if w.isalpha()])
    return stemmed_text
    

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
        return pr,rec
    
'''    
def check(chains, ch1,ch2):
    for c in chains:
        if ch1 in c and ch2 in c:
            return 2
    return 0

#exp params: 'Vectorizer', 'Spec', 'Treshold'
def evaluate(exp_params : dict,
             test_data_path = '../data/test_data.csv',
             df_true_path = '../data/toloka_result_pairs.csv',
             size = 300):
    
    
    test_table = pd.read_csv(test_data_path)
    df_true = pd.read_csv(df_true_path)
    interesting_df_true = df_true[df_true['res'] == 2]
        
    new1_set = set(interesting_df_true['new_1'].to_numpy())
    new2_set = set(interesting_df_true['new_2'].to_numpy())
        
    list_of_idx = np.random.choice(np.array(list(new1_set)),300) 
    list_of_idx2 = []
    for  i in list_of_idx:
        sub_df = interesting_df_true[interesting_df_true['new_1'] == i]
        sub_ar = sub_df['new_2'].to_numpy()
        c = np.random.choice(sub_ar)
        list_of_idx2.append(c)

    list_of_idx2 = np.array(list_of_idx2)
    final_list = np.concatenate((list_of_idx, list_of_idx2),axis = 0)
        
    reduced_data = test_table[test_table.index.isin(final_list)]
    
    myexp = MyExperiment(reduced_data, exp_params['Vectorizer'],
                         exp_params['Spec'],
                         exp_params['Treshold'])
    
    myexp.run()
    
    result_df1 = pd.DataFrame({'new_1' : [None] , 'new_2' : [None], 'res': [None]})
    
    N = len(list_of_idx)
    M = len(list_of_idx2)
    L1 = list_of_idx
    L2 = list_of_idx2

    for i in range(N):
        for j in range(M):
            result_df1.loc[j + i*M] = {'new_1' : L1[i], 'new_2' : L2[j],'res' :  check(myexp.result['Chains'], L1[i], L2[j])}
        print(i,'/',size)
    pr, rec = pairs_quality(df_true, result_df1)
    
    return pr, rec, myexp
    '''


















        
        
 
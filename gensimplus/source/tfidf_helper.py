#!/usr/bin/env python3
import os
import traceback
import pdb
import gensim
from gensimplus.source.gensim_plus_config import FLAGS
from gensimplus.source.model_save_load_helper import ModelSaveLoadHelper
from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim.models import TfidfModel
import jieba
import jieba.posseg
import re
from gensimplus.source.base_interface import TfidfHelperInterface
import numpy as np
import pdb

words2d = [
    ["富强","民主","文明","和谐","民主","文明","和谐"],
    ["自由","平等","公正","法治","法治"],
    ["爱国","敬业","明礼","诚信","敬业"]
]

words2d_eval = [
    ["富强","民主","文明"],
    ["自由","平等","公正"],
    ["爱国","敬业","敬业","敬业","发现"]
]

words = ["富强民主文明和谐", "自由平等公正法治", "爱国敬业明礼诚信"]

class TfidfHelper(TfidfHelperInterface):
    def __init__(self):
        pass
        print("\n> 实例化一个新的 TfidfHelper")

    def init_tfidf_model(self,dictionary,words2d):
        corpus = [dictionary.doc2bow(words) for words in words2d]
        tfidfModel = gensim.models.TfidfModel(corpus)
        modelSaveLoadHelperInstance = ModelSaveLoadHelper()
        modelSaveLoadHelperInstance.save(tfidfModel, "./tfidf.model")
        return tfidfModel
        #for corpus_item in corpus:
        #    #print([dictionary[i[0]] for i in corpus_item])
        #    yield [(dictionary[i[0]], i[1]) for i in tfidfModel[corpus_item]]

    def init_lsi_model(words2d,dictionary):
        corpus = [dictionary.doc2bow(words) for words in words2d]
        tfidfModel = gensim.models.TfidfModel(corpus)
        tfidf_corpus = tfidfModel[corpus]
        lsiModel = LsiModel(tfidf_corpus, dictionary)
        modelSaveLoadHelperInstance = ModelSaveLoadHelper()
        modelSaveLoadHelperInstance.save(tfidfModel, "./tfidf.model")

    def filter_tfidf(self,lst,percent,dictionary):
        """
        lst:2d数组词元素
        percent:tfidf重要度分位数
        """
        tfidf_sum_lst = [wordtuple[1] for wordtuple in lst]
        tfidf_sum_lst = list(set(tfidf_sum_lst))
        level = int(percent*len(tfidf_sum_lst))
        level_num = sorted(tfidf_sum_lst)[level]
        print(level,level_num,len(lst),lst)
        res = []
        for wordtuple in lst:
            if not wordtuple[1]<level_num:
                res.append(dictionary[wordtuple[0]])
        return res

    def test_init_tfidf_model(self):
        f = open("ll1.txt","r")
        lines = f.readlines()
        lines = [line.split("\t")[-1] for line in lines]
        word2d = [list(jieba.cut(line)) for line in lines]

        tfidfHelperInstance = TfidfHelper()
        dictionary = gensim.corpora.dictionary.Dictionary(word2d)
        tfidfmodel = tfidfHelperInstance.init_tfidf_model(dictionary,word2d)
        corpora_words2d = [dictionary.doc2bow(word1d) for word1d in word2d]
        doc_tfidf_lst = tfidfmodel[corpora_words2d]
        #corpora_words2d = [dictionary.doc2bow(words1d) for words1d in words2d_eval]
        for i in doc_tfidf_lst:
            res = self.filter_tfidf(i,0.5,dictionary)
            print(res)
            pdb.set_trace()

if __name__ == "__main__":
    tfidfHelperInstance = TfidfHelper()
    tfidfHelperInstance.test_init_tfidf_model()

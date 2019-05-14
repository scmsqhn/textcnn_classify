#!/usr/bin/env python3
import os
import traceback
import pdb
import gensim
from gensim_plus.source.gensim_plus_config import FLAGS
from gensim_plus.source.model_save_load_helper import ModelSaveLoadHelper
from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim.models import TfIdfModel
import jieba
import jieba.posseg
import re
from gensim_plus.source.base_interface import TfIdfHelperInterface
import numpy as np

words2d = [
    ["富强","民主","文明","和谐","民主","文明","和谐"],
    ["自由","平等","公正","法治","法治"],
    ["爱国","敬业","明礼","诚信","敬业"]
]

words = ["富强民主文明和谐", "自由平等公正法治", "爱国敬业明礼诚信"]

class TfIdfHelper(TfIdfHelperInterface):
    def __init__(self):
        pass
        print("\n> 实例化一个新的 TfIdfHelper")

    def init_tfidf_model(self,dictionary,words2d):
        corpus = [dictionary.doc2bow(words) for words in words2d]
        print(corpus)
        tfidfModel = gensim.models.TfidfModel(corpus)
        modelSaveLoadHelperInstance = ModelSaveLoadHelper()
        modelSaveLoadHelperInstance.save(tfidfModel, "../data/tfidf.model")
        for corpus_item in corpus:
            #print([dictionary[i[0]] for i in corpus_item])
            yield [(dictionary[i[0]], i[1]) for i in tfidfModel[corpus_item]]

    def init_lsi_model(words2d,dictionary):
        corpus = [dictionary.doc2bow(words) for words in words2d]
        tfidfModel = gensim.models.TfidfModel(corpus)
        tfidf_corpus = tfidfModel[corpus]
        lsiModel = LsiModel(tfidf_corpus, dictionary)
        modelSaveLoadHelperInstance = ModelSaveLoadHelper()
        modelSaveLoadHelperInstance.save(tfidfModel, "../data/tfidf.model")
        modelSaveLoadHelperInstance.save(tfidfModel, "../data/tfidf.model")

    def init_lsi_modl(self,common_corpus,dictionary):
        model = LsiModel(common_corpus, dictionary)
        vectorized_corpus = model[common_corpus]  # vectorize input copus in BoW format

    def filter_tfidf(self,lst,percent):
        tfidf_sum_lst = [wordtuple[1] for wordtuple in lst]
        level = int(percent*len(lst))
        level_num = sorted(tfidf_sum_lst)[level]
        res = []
        for wordtuple in lst:
            if wordtuple[1]>level_num:
                res.append(wordtuple[0])
        return res

    def test_init_tfidf_model(self):
        tfIdfHelperInstance = TfIdfHelper()
        dictionary = gensim.corpora.dictionary.Dictionary(words2d)
        gen = tfIdfHelperInstance.init_tfidf_model(dictionary,words2d)
        for i in range(len(words2d)):
            lst = gen.__next__()
            print(lst)
            res = tfIdfHelperInstance.filter_tfidf(lst,0.0)
            print(res)

if __name__ == "__main__":
    tfIdfHelperInstance = TfIdfHelper()
    tfIdfHelperInstance.test_init_tfidf_model()

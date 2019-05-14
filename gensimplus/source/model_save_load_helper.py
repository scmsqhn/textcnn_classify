#!/usr/bin/env python3
import os
import traceback
import pdb
import gensim
from gensimplus.source.gensim_plus_config import FLAGS
import jieba
import jieba.posseg
import re
from gensimplus.source.base_interface import Word2VecHelperInterface
from gensimplus.source.base_interface import ModelSaveLoadHelperInterface
import numpy as np

class ModelSaveLoadHelper(ModelSaveLoadHelperInterface):

    def __init__(self):
        pass
        print("\n> 实例化一个新的 ModelSaveLoadHelper")

    def save(self,model,path):
        model.save(path)

    def load(self,path,model_type):
        self.fetch_modelswitch_model_type(path,model_type)

    def fetch_model_with_type(self,path,model_type):
        model_name_func_dct = {
            FLAGS.tfidf_model:  lambda : gensim.models.tfidf.load(path),
            FLAGS.lsi_model:    lambda : gensim.models.lsi.load(path),
            FLAGS.lda_model:    lambda : gensim.models.lda.load(path)
        }
        return model_name_func_dct[model_type](path)



if __name__ == "__main__":
    modelSaveLoadHelperInstance = ModelSaveLoadHelper()


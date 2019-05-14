#!/usr/bin/env python3
import gensim
import os
import pdb
from gensimplus.source.gensim_plus_config import FLAGS
import traceback
from gensimplus.source.base_interface import DictionaryHelperInterface

class DictionaryHelper(DictionaryHelperInterface):

    def __init__(self):
        pass

    def newDictionary(self, wordsLst2d):
        """
        des: 使用新的文本生成词典
        in: 二维数组,元素是词
        out: 词典
        """
        try:
            dct = gensim.corpora.dictionary.Dictionary(wordsLst2d)
            return dct
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def addDoc2Dct(self, wordLst2d, dct):
        """
        使用新的文本生成词典，与现有词典合并
        """
        try:
            dct.adddocuments(wordLst2d)
            return dct
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def loadDct(self, dctPath):
        try:
            dct = gensim.corpora.dictionary.Dictionary()
            return dct.load(dctPath)
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def savDct(self, dctPath, dct):
        try:
            return dct.save(dctPath)
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def char2id(self,dct,char):
        try:
            return dct.token2id[char]
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def id2char(self,dct,_id):
        try:
            return dct.get(_id)
        except:
            traceback.print_exc()
            return FLAGS.ERROR

def func_test():
    words2dLst =[["我","爱","北京","天安门"],["我","爱","万里","长城"]]
    dctHelperIns  = DictionaryHelper()
    dct = dctHelperIns.newDictionary(words2dLst)
    print("\n> 新建词典完成",dct)
    _ = dctHelperIns.savDct(os.path.join(".","dct"),dct)
    print("\n> 保存词典完成",_)
    dct = dctHelperIns.loadDct("./dct")
    #pdb.set_trace()
    print("\n> 加载词典完成",dct)
    _ = dctHelperIns.char2id(dct,"北京")
    print("\n> 词转id",_)
    _ = dctHelperIns.id2char(dct,1)
    print("\n> id转词",_)

if __name__ == "__main__":
    func_test()



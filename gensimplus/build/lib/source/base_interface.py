#!/usr/bin/env python3
from abc import ABCMeta,abstractmethod

class Word2VecHelperInterface(metaclass=ABCMeta):
    """
    该类别,类似与java中的interface
    接口方法在子类中,必须实现
    """
    @abstractmethod
    def setFileInPath(self,dataFileInPath):
        """
        设置输入数据路径,读取数据,均来自该目录
        """
        pass

    @abstractmethod
    def setFileOutPath(self,dateFileOutPath):
        """
        设置目标输出数据路径,写入数据,到该目录
        """
        pass

    @abstractmethod
    def setDataBasePath(self, dataBaseParameter):
        """
        设置数据库参数
        """
        pass

    @abstractmethod
    def trainWord2VecModel(self, paraDct):
        """
        训练一个词向量模型
        """
        pass


class DictionaryHelperInterface(metaclass=ABCMeta):
    """
    该类别,类似与java中的interface
    接口方法在子类中,必须实现
    """
    @abstractmethod
    def newDictionary(self, wordsLst2d):
        """
        新建词典
        """
        pass

    @abstractmethod
    def addDoc2Dct(self, wordLst2d, dct):
        """
        添加文本以补充新的词进入词典
        """
        pass

    @abstractmethod
    def loadDct(self, dctPath):
        """
        加载词典
        """
        pass

    @abstractmethod
    def savDct(self, dctPath, dct):
        """
        保存词典
        """
        pass

    @abstractmethod
    def char2id(self,dct,char):
        """
        字转id
        """

    @abstractmethod
    def id2char(self,dct,char):
        """
        保存词典
        """
        pass

class TfIdfHelperInterface(metaclass=ABCMeta):

    @abstractmethod
    def init_tfidf_model(self,dictionary,words2d):
        """
        新建tfidf模型
        """
        pass

    @abstractmethod
    def filter_tfidf(self,lst,percent):
        pass


class ModelSaveLoadHelperInterface(metaclass=ABCMeta):

    @abstractmethod
    def save(self,model,path,model_type):
        """
        保存模型
        """

    @abstractmethod
    def load(self,path,model_type):
        """
        加载模型
        """


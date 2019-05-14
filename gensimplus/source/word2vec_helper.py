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
import numpy as np

class Word2VecHelper(Word2VecHelperInterface):

    def __init__(self):
        pass
        print("\n> 实例化一个新的 Word2VecHelper")

    """
    def init_word_flag(self):
        self.nous = self.FLAGS.nous.split(",")
        self.adj = self.FLAGS.adj.split(",")
        self.pj = self.FLAGS.pj.split(",")
        self.verb = self.FLAGS.verb.split(",")
    """

    def setFileInPath(self,dataFileInPath):
        self.dataFileInPath = dataFileInPath

    def setFileOutPath(self,dataFileOutPath):
        self.dataFileOutPath = dataFileOutPath

    def setDataBasePath(self, dataBaseParameter):
        """
        dataBaseParameter是一个dict类型的变量用于初始化mysql的连接
        """
        self.dataBaseParameter = dataBaseParameter

    def test_trainWord2VecModel(self):
        doc = ["中华人民共和国中央人民政府今天成立了","富强民主文明和谐,自由平等公正法制,爱国敬业明理诚信"]
        w2v_model = self.trainWord2VecModel(doc,FLAGS.word2vec_4_word,128,0)
        if FLAGS.DEBUG == "True":
            pdb.set_trace()
            print(w2v_model)
        return w2v_model

    def trainWord2VecModel(self, doc, flag=FLAGS.word2vec_4_word, num_class=128,min_count=5):
        """
        docin:一维数组,元素为文本

        ex of in:['今天上午','明天下午']
        out:Word2Vec model
        """
        try:
            words_2d, flags_2d,chars_2d = [],[],[]
            if flag == FLAGS.word2vec_4_word:
                for sentence in doc:
                    word_flag_tuple = jieba.posseg.cut(sentence)
                    words = [item.word for item in word_flag_tuple]
                    words_2d.append(words)
                return gensim.models.word2vec.Word2Vec(words_2d, size=num_class, min_count=min_count)
            elif flag == FLAGS.word2vec_4_word_complex:
                for sentences in doc:
                    word_flag_tuple = jieba.posseg.cut(sentence)
                    words = [item.word for item in word_flag_tuple]
                    flags = [item.flag for item in word_flag_tuple]
                    word_flag_str = "".join(["%s/%s "%(i,j) for i,j in zip(words,flags)])
                    n_wd = re.findall("[^ ]+/(%s) "%FLAGS.nous)
                    v_wd = re.findall("[^ ]+/(%s) "%FLAGS.verb)
                    adj_wd = re.findall("[^ ]+/(%s) "%FLAGS.adj)
                    p_wd = re.findall("[^ ]+/(%s) "%FLAGS.pj)
                    words.extend(n_wd)
                    words.extend(v_wd)
                    words.extend(adj_wd)
                    words.extend(p_wd)
                    words.extend(list(sentences))
                    words.extend(flags)
                    words_2d.append(words)
                return gensim.models.word2vec.Word2Vec(words_2d, num_class=num_class, min_count=min_count)
            elif flag == FLAGS.word2vec_char:
                for sentence in doc:
                    words_2d.append(list(sentence))
                return gensim.models.word2vec.Word2Vec(words_2d, num_class=num_class, min_count=min_count)
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def loadWord2VecModel(self, word2vecModelPath):
        try:
            model = gensim.models.word2vec.Word2Vec.load(word2vecModelPath)
            return model
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def saveWord2VecModel(self, word2vecModelPath, model):
        try:
            model.save(word2vecModelPath)
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def getIdsFromModel(self,wordsLst,model,dummy,size):
        """
        in:
            wordsLst:一维数组
            model:模型
            dummy:填充字符
            size:batch长度
        out:
            一维数组,元素numpy.array([,128])
        """
        try:
            res = []
            if len(wordsLst)<size:
                wordsLst.extend([dummy]*size)
                res = wordsLst[:size]
            else:
                res = wordsLst[:size]
                print(res)
                print("\n> 有部分文本被丢弃,数量为 %s ;\n> 内容为 %s ;"%(len(wordsLst)-size,wordsLst[size:]))
            _ = np.concatenate(model[res]).reshape(size,128)
            if FLAGS.DEBUG == "True":
                pdb.set_trace()
            return _
        except:
            traceback.print_exc()
            return FLAGS.ERROR

    def test_getBatchFromModel(self):
        doc = [["自由","平等"],["富强","民主"]]
        model = self.loadWord2VecModel(os.path.join(FLAGS.current_path,"word2vec.model"))
        batch = self.getBatchFromModel(doc,model,"富强",400)
        if FLAGS.DEBUG == "True":
            pdb.set_trace()
        print(batch)

    def getBatchFromModel(self,doc,model,dummy,size):
        """
        in:
            doc:一维数组,元素string
            model:模型
            dummy:填充字符
            size:batch长度
        out:
            二维数组,元素numpy.array([,128])
        """
        try:
            res = []
            for sentence in doc:
                print(sentence)
                _ = self.getIdsFromModel(sentence,model,dummy,size)
                print(_)
                res.append(_)
            out = np.concatenate(res).reshape(len(doc),size,128)
            if FLAGS.DEBUG == "True":
                print(out,out.shape)
            return out
        except:
            traceback.print_exc()

if __name__ == "__main__":
    word2vechelper_instance = Word2VecHelper()
    model = word2vechelper_instance.test_trainWord2VecModel()
    word2vechelper_instance.saveWord2VecModel(os.path.join(FLAGS.current_path,"word2vec.model"), model)
    word2vechelper_instance.test_getBatchFromModel()


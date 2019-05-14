#!/usr/bin/env python3
import sys
sys.path.append("/home/distdev")
import gensimplus
from gensimplus.source.tfidf_helper import TfidfHelper
import traceback
import pymongo
import logging
import collections
import gensimplus
import sys
from sklearn.manifold import TSNE
#from bilstm import addr_classify
#from bilstm import eval_bilstm
import pdb
import arctic
import os
import pdb
import pdb
#import pdb
import gensim
import traceback
#import digital_info_extract as dex
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
#import time
#import os
import jieba
import jieba.posseg as seg
import re
lennum = 1000
def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    #_print("\n> CURPATH IS ", CURPATH)
    return os.path.join(CURPATH, filepath)

dctpath = _path("data/all_addr_dict.txt")
#jieba.load_userdict(dctpath)
import jieba.posseg as pseg
#import collections
#import sklearn.utils
#from sklearn.utils import shuffle
#import myconfig as config
#import tensorflow as tf
#
#from addr_classify import Addr_Classify
import sys
import const
Const = const._const()
Const.__setattr__("SUCC", "\n> success")
Const.__setattr__("FAIL", "\n> fail")
Const.__setattr__("ERROR", "\n> error")
Const.__setattr__("TEXTUSELESS", "\n无效原文 continue")
Const.__setattr__("TARGETUSELESS", "\n无效目标词 continue")
Const.__setattr__("KEYLOSS", "\n无该key continue")
Const.__setattr__("CLASSIFY_BATCH", "\n输出分类样本batch")
Const.__setattr__("DICT_LOST", "\n该词语在词典中并不存在")
Const.__setattr__("DEBUG",False)
Const.__setattr__("FLAG",False)
Const.__setattr__("NOUS",['n','s'])
Const.__setattr__("NS",['ns'])
Const.__setattr__("NZ",['nz'])
Const.__setattr__("NR",['nr'])
Const.__setattr__("PJ",['p'])
Const.__setattr__("AD",['ad','d','a'])
Const.__setattr__("V",['v','vn'])
Const.__setattr__("VP",['v','vn','n'])
#Const.__setattr__("VP",['v','vn','ad','d','a','n','ns','nz','nr'])
Const.__setattr__("NUM",['m'])
Const.__setattr__("ENG",['eng'])
Const.__setattr__("X",['x'])

Const.str2var()

global SAMPLE_CNT
global DCTFLAG
DCTFLAG = True#False#True
SAMPLE_CNT = set()

"""
STOP_WORD = []
with open("./stop_word.txt","r")as f:
    lines  = f.readlines()
    for line in lines:
        STOP_WORD.append(line)
print(STOP_WORD[:2])
##"""

def logging_init(filename="./logger.log"):
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

lgr = logging_init()

pred_lgr = logging_init(filename="./eval.log")

def _print_pred(lgrnm=pred_lgr, *l):
    logger = lgrnm
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

def _print(*l):
    logger = lgr
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

tran_prob = {'06': 0.00011000110001100011,\
 '11': 0.00016000160001600016,\
 '12': 0.02900029000290003,\
 '13': 0.24118241182411823,\
 '14': 0.00012000120001200014,\
 '21': 3.000010000100003e-05,\
 '22': 0.014580145801458014,\
 '23': 0.02895028950289503,\
 '24': 2.000020000200002e-05,\
 '31': 0.2108521085210852,\
 '34': 0.05918059180591806,\
 '35': 9.000090000900009e-05,\
 '37': 1.000010000100001e-05,\
 '41': 0.05886058860588606,\
 '44': 0.08012080120801209,\
 '45': 0.012880128801288013,\
 '47': 2.000020000200002e-05,\
 '50': 0.00011000110001100011,\
 '51': 5.000050000500005e-05,\
 '54': 0.0001000010000100001,\
 '56': 0.1128511285112851,\
 '61': 0.00025000250002500023,\
 '64': 0.012230122301223013,\
 '65': 0.08741087410874109,\
 '67': 0.013070130701307013,\
 '71': 0.0002200022000220003,\
 '74': 9.000090000900009e-05,\
 '75': 0.012730127301273013,\
 '77': 0.024640246402464025}


class Data_Helper(object):

    def __init__(self):
        _print("\ncls Data_Helper instance")
        #assert self.arctic_inf_init() == Const.SUCC
        #self.mongo_inf_init("myDB", "gz_gongan_case")
        self.odd= True
        #self.w2vm = bilstm.w2vm.load_w2vm()
        self.btsize=64
        #self.mongo_inf_init("myDB", "gz_gongan_alarm_1617")
        #self.w2vm = gensim.models.word2vec.Word2Vec.load("/home/distdev/bilstm/model/w2vm")
        #self.dct = gensim.corpora.Dictionary.load("./model/myDctBak")
        #self.ac =addr_classify.Addr_Classify(["2016年1月1日9时左右，报警人文群华在股市云岩区保利云山国际13栋1楼冬冬小区超市被撬门进入超市盗走现金1200元及一些食品等物品。技术科民警已经出现场勘查。"])
        self.train_data_generator = self.gen_train_data('train')
        self.eval_data_generator = self.gen_train_data("eval")
        #self.tags = {'x':0.0, 'o':1.0,'a':2.0,'r':3.0,'v':4.0,'d':5.0}
        self.tags = {'o':0,'b':2,'i':1}# words bg mid end / addrs bg mid end
        stw = ['了','客户','表示','我行','处理','要求','谢谢','贵行','烦请','的']
        f = open(_path("stw.txt"),"r")
        pass # print('mk')
        stwwords_4 = f.read().split('\n')
        pass # print('mk')
        #stwwords_0, stwwords_1, stwwords_2, stwwords_3 = self.stopwords()
        pass # print('mk')
        #stwwords_3.extend(stwwords_4)
        pass # print('mk')
        #stw.extend(stwwords_2)
        pass # print('mk')
        #for i in [stwwords_0, stwwords_1, stwwords_2, stwwords_3]:
        #    if " " in i:
        #        i.remove(" ")
        #self.stw0 = set(stwwords_0)
        #self.stw1 = set(stwwords_1)
        #self.stw2 = set(stwwords_2)
        #self.stw3 = set(stwwords_3)
        #self.stw4 = set(stwwords_4)
        #self.stw.remove(" ")
        #print(len(self.stw))
        self.w2vmodel = gensim.models.word2vec.Word2Vec.load(_path("w2vmodel"))
        self.tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=1000,method='exact')
        self.c2n,self.n2c=self.get_lb()
        self.arctic_inf_init()
        self.trans_lb()
        #self.genDct()
        self.dct = gensim.corpora.Dictionary.load(_path("./dictionary.dct"))


        #self.tags = {'o':0,'b':1,'i':2,'e':3,'s':4,'a':5,'d':6,'r':7,'v':8}# words bg mid end / addrs bg mid end


    def trans_lb(self):
        LBDCT={}
        with open(_path("lbmap.txt"),'r') as  f:
            lines = f.readlines()
            for line in lines:
                pass
                kv = line.split("\t")
                LBDCT[kv[0]] = re.sub("\n","",kv[1])
        self.LBDCT = LBDCT
        print(self.LBDCT)

    def stopwords(self):
        #w2v = gensim.models.word2vec.Word2Vec()
        f = open(_path("../labelmarker/data/five_classify_train_250k.txt"),"r")
        g = open(_path("../labelmarker/data/five_classify_eval.txt"),"r")
        ls = f.readlines()
        gs = g.readlines()
        ls.extend(gs)
        sents = []
        for i in ls:
            sents.extend(list(jieba.cut(self.dwc(i))))
        dct = dict(collections.Counter(sents))
        stw_lv1 = []
        stw_lv2 = []
        stw_lv3 = []
        stw_lv4 = []
        for i in dct.keys():
            if dct[i]<4:
                stw_lv1.append(i)
            elif dct[i]<20:
                continue#stw.append(i)
            elif dct[i]<lennum:
                stw_lv2.append(i)
            elif dct[i]<10000:
                stw_lv3.append(i)
            else:
                stw_lv4.append(i)
        return stw_lv1, stw_lv2, stw_lv3, stw_lv4

    def vec(self, nwd, vwd, pwd, words, nums, chars, markers, dummy):
      if Const.FLAG:
        pdb.set_trace()
      try:
        printwds = []
        stpwds0 = []
        stpwds1 = []
        stpwds2 = []
        stpwds3 = []
        for word in words:
            #print(word)
            if word == ",":
                continue
            if word in self.stw0:
                stpwds0.append(word)
            elif word in self.stw1:
                stpwds1.append(word)
            elif word in self.stw2:
                stpwds2.append(word)
            elif word in self.stw3:
                stpwds3.append(word)
            else:
                printwds.append(word)
        res = np.array([])
        for word in printwds:
            try:
                #print('mk')
                w2v = self.w2vmodel[word]
            except:
                pass # print(word)
                pass # print('ex')
                continue
            res = np.concatenate([res, w2v])
        l = (12200-res.shape[0])//128
        print(l)
        if l<0:
            res = res[:12200]
        else:
          for _ in range(l):
            #print('dummk')
            res = np.concatenate([res, dummy])
        assert (12200-res.shape[0])%128 == 0

        print(l)
        for word in stpwds0:
            try:
                w2v = self.w2vmodel[word]
            except:
                pass # print(word)
                pass # print('ex')
                continue
            res = np.concatenate([res, w2v])

        l = (25600-res.shape[0])//128
        if l<0:
            res = res[:25600]
        else:
          for _ in range(l):
            #print('dummk')
            res = np.concatenate([res, dummy])

        assert (25600-res.shape[0])%128 == 0
        assert res.shape == (25600,)

        print(l)
        print(nwd)
        for words in nwd:
            if word in self.stw0:
                continue
            _ = self.w2vmodel[word]
            print(word)
            res = np.concatenate([res, _])
        assert (12200*3-res.shape[0])%128 == 0
        l = (12200*3-res.shape[0])//128
        if l<0:
            res = res[:12200*3]
        else:
          for _ in range(l):
            #print('dummk')
            res = np.concatenate([res, dummy])

        print(vwd)
        for word in vwd:
            if word in self.stw0:
                continue
            _ = self.w2vmodel[word]
            res = np.concatenate([res, _])
        assert (3.5*12200-res.shape[0])%128 == 0
        l = (12200*3.5-res.shape[0])//128
        print(l)
        if l<0:
            res = res[:int(12200*3.5)]
        else:
          for _ in range(int(l)):
            #print('dummk')
            res = np.concatenate([res, dummy])

        print(pwd)
        for word in pwd:
            if word in self.stw0:
                continue
            _ = self.w2vmodel[word]
            res = np.concatenate([res, _])
        assert (12200*4-res.shape[0])%128 == 0
        l = (12200*4-res.shape[0])//128
        print(l)
        if l<0:
            res = res[:12200*4]
        else:
          for _ in range(l):
            #print('dummk')
            res = np.concatenate([res, dummy])

        for num in range(40):
            if num < len(nums):
                _ = self.w2vmodel[nums[num]]
                res = np.concatenate([res, _])
            else:
                res = np.concatenate([res, dummy])

        for char in range(30):
            if char < len(chars):
                _ = self.w2vmodel[chars[char]]
                res = np.concatenate([res, _])
            else:
                res = np.concatenate([res, dummy])

        for marker in range(30):
            if marker < len(markers):
                _ = self.w2vmodel[markers[marker]]
                res = np.concatenate([res, _])
            else:
                res = np.concatenate([res, dummy])

        for word in stpwds1:
            try:
                w2v = self.w2vmodel[word]
            except:
                pass # print(word)
                pass # print('ex')
                continue
            res = np.concatenate([res, w2v])
        l = (12200*5+6200-res.shape[0])//128
        if l<0:
            res = res[:12200*5+6200]
        else:
          for _ in range(l):
            res = np.concatenate([res, dummy])
            #print('dummk')

        for word in stpwds2:
            try:
                w2v = self.w2vmodel[word]
            except:
                pass # print(word)
                pass # print('ex')
                continue
            res = np.concatenate([res, w2v])
        l = (12200*6-res.shape[0])//128
        if l<0:
            res = res[:12200*6]
        else:
          for _ in range(l):
            res = np.concatenate([res, dummy])
            #print('dummk')

        for word in stpwds3:
            try:
                w2v = self.w2vmodel[word]
            except:
                pass # print(word)
                pass # print('ex')
                continue
            res = np.concatenate([res, w2v])
        l = (12200*6+6200-res.shape[0])//128
        if l<0:
            res = res[:12200*6+6200]
        else:
          for _ in range(l):
            res = np.concatenate([res, dummy])
            #print('dummk')
        assert (12200*6+6200-res.shape[0]) == 0
        print(l)
        print(res.shape)
        if Const.FLAG:
            pdb.set_trace()
        return res
      except:
        traceback.print_exc()

    def vecs(self,words):
        return self.w2vmodel[words]


    def common_data_prepare(self):
        """
        prepare the data of common, before train
        """
        pass

    #def mongo_inf_init(self, lib_nm, col_nm):
    #    # handle the mongo db data
    #    self.conn = pymongo.MongoClient("mongodb://127.0.0.1")
    #    self.get_mongo_coll(lib_nm, col_nm)
    #    return Const.SUCC

    #def get_mongo_coll(self, lib_nm, col_nm):
    #    self.mongo_lib = self.conn[lib_nm]
    #    self.mongo_col = self.mongo_lib[col_nm]
    #    return self.mongo_col#

    def arctic_inf_init(self):
        # handle the pd data, like panel dataframe series
        # with arctic to hadnle mongodb local
        self.store = arctic.Arctic('mongodb://127.0.0.1')
        return Const.SUCC

    def sav2Arctic(self, ndarr, name):
        store = self.store
        store.initialize_library('jh')
        library = store['jh']
        #aapl = quandl.get("WIKI/AAPL", authtoken="your token here")
        # Store the data in the library
        library.write(name, ndarr, metadata={'source': name})


    def loadFromArctic(self, name):
        store = self.store
        store.initialize_library('jh')
        library = store['jh']
        #aapl = quandl.get("WIKI/AAPL", authtoken="your token here")
        item = library.read(name)
        return  item.data
        #metadata = item.metadata


    def arctic_get_all(self):
        # handle the pd data, like panel dataframe series
        libs_name = self.store.list_libraries()
        for lib_name in libs_name:
            self.arc_libs[lib_name] = self.store[lib_name]
        for _lib in self.arc_libs.keys():
            colls_name = self.arc_libs[_lib].list_symbols()
            for col_name in colls_name:
                if col_name in self.arc_colls.keys():
                    _print("\n there r two colls has the same name, the old one is be cover, modify the coll name or use lib.coll to request")
                _print(_lib)
                self.arc_colls[col_name] = self.store[_lib].read(col_name).data
        _print("\n there r %s libs and %s colls"%(len(self.lib), len(self.arc_colls)))
        return Const.SUCC

    def get_arctic_df(self, lib_nm, col_nm):
        collection = self.store[lib_nm].read(col_nm)
        #cnt = collection.data.count()
        dat = collection.data
        return dat

    def marker_sentence(self, txt):#bie s
        words_marker = ""
        words = list(jieba.cut(txt))
        for word in words:
            #if word in STOP_WORD:
            #    continue
            if word == '\r':
                words_marker+=word
            elif word == '\n':
                words_marker+=word
            elif len(word) == 1:
                _ = "%(.+?)/(.) "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%(.+?)/(.) %(.+?)/(.) "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%(.+?)/(.) "%word[0]
                for i in range(1, len(word)-2):
                    _+="%(.+?)/(.) "%word[i]
                _+="%(.+?)/(.) "%word[-1]
                words_marker+=_
        return words_marker


    def marker_target_pre_aft(self, txt, cont):#avd r
        res = []
        for word in txt:
            l = len(word)
            index = cont.find(word)
            index_lst = [index-2,index-1,index+l+1,index+l+2]
            res.append(index_lst)
        return res

    def marker_target(self, txt):#avd r
        words_marker = ""
        words = list(jieba.cut(txt))
        for word in words:
            #if word in STOP_WORD:
            #    continue
            if word == '\r':
                words_marker+=(word+"/s" )
            elif word == '\n':
                words_marker+=(word+"/s" )
            elif len(word) == 1:
                _ = "%(.+?)/(.) "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%(.+?)/(.) %(.+?)/(.) "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%(.+?)/(.) "%word[0]
                for i in range(1, len(word)-2):
                    _+="%(.+?)/(.) "%word[i]
                _+="%(.+?)/(.) "%word[-1]
                words_marker+=_
        return words_marker

    def textcnn_data_transform(self, data, n):
        assert n%2 ==1
        m = n//2
        """
        input data is a (1000,9) array
        """
        assert data.shape == (1000,9)
        output = []
        for i in range(0,1000):
            for j in range(i-m,i+m):
                if j<0 or j>999:
                    output.extend([0.0]*8)
                else:
                    output.extend(data[j,:])
        pass#pdb.set_trace()
        #print(np.array(output).reshape(1000,9*(n-1)))
        return np.array(output).reshape(1000,9*(n-1))

    def gen_train_data_beta(self, name="train"):
        self.get_mongo_coll('myDB','traindata')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "traindata")
        _cursor = _collections.find()
        _count = _collections.count()
        begin_cursor,end_cursor = -1,-1
        if name == "train":
            begin_cursor = 100
            end_cursor = _count
        elif name == 'eval':
            begin_cursor = 0
            end_cursor = 100
        _print("begin_cursor, end_cursor")
        _print(begin_cursor)
        _print(end_cursor)
        ll = [i for i in range(begin_cursor, end_cursor)]
        np.random.shuffle(ll)
        for c in ll:
            _crim = _cursor[c]['addrcrim_sum']
            if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1:
                continue
            i = _cursor[c]
            try:
                len(i['addrcrim'])
            except:
                pass # print(i['addrcrim'])
                continue
            if type(_crim) == str:
                _crim = _crim.split(",")
            for _ in _crim.copy():
                if len(_)<3:
                   _crim.remove(_)
            _text = ""
            try:
                _text = _cursor[c]['text']
            except KeyError:
                traceback.print_exc()
            #print(_text)
            _text = self.dwc(_text)
            #flag = False
            if len(_text)<3:
                continue
            txt = _text
            cri = []
            cri = [self.dwc(i) for i in _crim]
            if len(cri)==0:
                continue
            _print(cri)
            #mark_sent = self.marker_sentence(txt)
            words_markers = self.marker_target_pre_aft(cri,  txt)
            a = [i[0] for i in words_markers]
            b = [i[1] for i in words_markers]
            c = [i[2] for i in words_markers]
            d = [i[3] for i in words_markers]
            mark_target_sent = ""
            for i in range(len(txt)):
                if txt[i] == "\r":
                    mark_target_sent += "\r"
                elif txt[i] == "\n":
                    mark_target_sent += "\n"
                elif i in a:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in b:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in c:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in d:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                else:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
            mark_sent_lst = mark_target_sent.split("\n")
            self.get_mongo_coll('myDB','train_data_bio').insert({"text":mark_target_sent})
            _print("\n> mark_sent_lst: ", mark_sent_lst)
            ids_lst, tags_lst = [], []
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def gen_train_data_arf(self, name="train"):
        self.get_mongo_coll('myDB','traindata')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "traindata")
        _cursor = _collections.find()
        _count = _collections.count()
        begin_cursor,end_cursor = -1,-1
        if name == "train":
            begin_cursor = 100
            end_cursor = _count
        elif name == 'eval':
            begin_cursor = 0
            end_cursor = 100
        _print("begin_cursor, end_cursor")
        _print(begin_cursor)
        _print(end_cursor)
        ll = [i for i in range(begin_cursor, end_cursor)]
        np.random.shuffle(ll)
        for c in ll:
            _crim = _cursor[c]['addrcrim_sum']
            if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1:
                continue
            i = _cursor[c]
            try:
                len(i['addrcrim'])
            except:
                pass # print(i['addrcrim'])
                continue
            if type(_crim) == str:
                _crim = _crim.split(",")
            for _ in _crim.copy():
                if len(_)<3:
                   _crim.remove(_)
            _text = ""
            try:
                _text = _cursor[c]['text']
            except KeyError:
                traceback.print_exc()
            #print(_text)
            _text = self.dwc(_text)
            #flag = False
            if len(_text)<3:
                continue
            txt = _text
            cri = []
            cri = [self.dwc(i) for i in _crim]
            if len(cri)==0:
                continue
            _print(cri)
            mark_sent = self.marker_sentence(txt)
            for add_cri in cri:
                if len(add_cri)<2:
                    continue
                lgr.debug('add_cri')
                lgr.debug(add_cri, type(add_cri))
                mark_target = self.marker_target(add_cri)
                mark_target_sent = self.marker_sentence(add_cri)
                #_print(mark_target)
                #_print(mark_target_sent)
                try:
                    mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
                except:
                    pass#pdb.set_trace()
                    pass
                mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
            #_print("\n> mark_sent")
            #_print(mark_sent)
            pass#pdb.set_trace()
            #_print(re.findall("(.+?)/(.+?) ", mark_sent))
            #mark_sent_lst = mark_sent.split("\n")
            ids_lst, tags_lst = [], []
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def marker(self, text, flag):
        result = ""
        if len(text)=="":
            return result
        if flag:
            pass # print(text)
            words = self.clr_2_lst(text)
            result += "%s/b "%words[0]
            for word in words[1:]:
                result += "%s/i "%word
        else:
            words = self.clr_2_lst(text)
            for word in words:
                result += "%s/o "%word
        pass # print("\n",text,"\n",result)
        return result

    def split(self, lst, text):
        result = ""
        target = []

        regex1 = ""
        for word in lst:
            _ = "(.*?)"+word
            regex1+=_
        regex1+="(.*?)$"

        regex2 = ""
        for word in lst[::-1]:
            _ = "(.*?)"+word
            regex2+=_
        regex2+="(.*?)$"
        pass # print(self.dwc(text))
        pass # print(regex1, regex2)
        try:
            target1 = list(re.findall(regex1, self.dwc(text))[0])
        except:
            target1 = []
            #traceback.print_exc()
        try:
            target2 = list(re.findall(regex2, self.dwc(text))[0])
        except:
            target2 = []
            #traceback.print_exc()
        pass # print(target)
        if len(target1)>0:
            target = target1
            lst=lst
        elif len(target2)>0:
            target = target2
            lst=lst[::-1]
        else:
            return result
        assert len(target) == len(lst)+1
        for i,j in zip(target[:-1],lst):
            result+=self.marker(i, False)
            result+=self.marker(j, True)
        result+=self.marker(target[-1], False)
        pass # print("result", result)
        pass # print("text", text)
        pass # print("lst",lst)
        pass # print("target",target)
        import pdb
        pass#pdb.set_trace()
        return result


    def clr_2_lst(self,sent):
        pass # print(sent)
        res = [self.dwc(i) for i in list(jieba.cut(sent))]
        return res

    def clr_2_str(self,sent):
        pass
        sent = self.dwc(sent)
        res = ""
        for i in list(jieba.cut(sent)):
            res+=i
        return res

    def stopsub(self, sent):
        sent__ = sent
        for i in self.stw:
            sent__ = re.sub(i,"",sent__)
        return sent__

    def ndwc(self,sent):
        sent = re.sub("[^0-9]","",sent) # marker
        return list(sent)

    def cdwc(self,sent):
        sent = re.sub("[^a-zA-Z]","",sent) # marker
        return list(sent)

    def mdwc(self,sent):
        sent = re.sub("[\u4e00-\u9fa5a-zA-Z0-9]","",sent) # marker
        return list(sent)

    def dwc(self,sent):
        #sent = self.stopsub(sent)
        sent = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9]",",",sent) # marker
        #sent = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9]","",sent) # marker
        #sent = re.sub("[^\u4e00-\u9fa5a-z0-9A-Z@\.]","",sent) # marker
        return sent

    def gen_train_data(self, name="train"):
      g = ""
      if name == "train":
          g = self.gen_train(begin_cursor=100, end_cursor=-1)
      elif name =="eval":
          g = self.gen_train(db='myDB',coll='traindata',textcol='text',targetcol='addrcrim',funcname='gen_train',begin_cursor=0, end_cursor=100)
          #g = self.gen_eval(funcname="gen_eval",columns_name="text",columns_name_tar="addrcrim",db="myDB",coll="traindata",begin_cursor=0,end_cursor=100)
      elif name =="evalTaiyuan":
          g = self.gen_eval(funcname="gen_eval",columns_name="casdetail",columns_name_tar="",db="myDB",coll="original_data",begin_cursor=0,end_cursor=100, wordFilter=False)
          #gen = datasrc.gen_eval(funcname="gen_eval",columns_name="text",columns_name_tar="addrcrim",db="myDB",coll="traindata", begin_cursor=0, end_cursor=100)
      else:
          pass
      return g

    def format_str(self, *para):
        strout=""
        lenth = len(para)
        for i in range(lenth):
            strout+="{%s},"%i
        return strout[:-1].format(*para)

    def random_lst(self, ll):
        np.random.shuffle(ll)
        return ll

    def throw_exception(self, sent):
        raise Exception(sent)

    def _vali_type(self,dat,tp,name): # (dat) type should be (tp)
        try:
            assert type(dat) == tp
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> the type of',dat,'!=equal',tp)
            self.throw_exception(sent)
            return Const.ERROR

    def _vali_equal(self,left,right,relation,name): # left right is equal small or big
        try:
            if relation=="==":
                assert left==right
            elif relation==">":
                assert left>right
            elif relation=="<":
                assert left<right
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> %s and %s is not %s '%(left,right,relation))
            self.throw_exception(sent)
            return Const.ERROR

    def _vali_in(self,child,parent,name): # left right is equal small or big
        try:
            if type(parent)==dict:
                assert (child in parent) ==True
            elif type(parent)==list:
                assert (child in parent) ==True
            elif type(parent)==tuple:
                assert (child in parent) ==True
            elif type(parent)==set:
                assert (child in parent) ==True
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> %s is not in %s '%(child, parent))
            self.throw_exception(sent)
            return Const.ERROR

    def _vali_date_lenth(self,dat,lenth,name): # (dat) type should be (tp)
        try:
            assert type(dat) == list or tuple
        except AssertionError:
            sent = self.format_str('\n>In function',name,'\n> the type of', dat,' has no function len(), only list and tuple has lenth')
            self.throw_exception(sent)
            return Const.ERROR
        try:
            assert len(dat) == lenth
            return Const.SUCC
        except AssertionError:
            sent = self.format_str('\n>In function',name,'\n> the lenth of', dat,'!=equal',lenth)
            self.throw_exception(sent)
            return Const.ERROR

    def toLst(self, s):
        if type(s)==list:
            return s
        elif type(s)==str:
            return s.split(",")
        elif type(s)==tuple:
            return list(s)

    def toStr(self, s):
        if type(s)==list:
            return ",".join(s)
        elif type(s)==str:
            return s
        elif type(s)==tuple:
            return ",".join(list(s))

    def sentFromDct(self, sent):
        ids = []
        words = list(jieba.cut(sent))
        for word in words:
            _id = self.fromdct(word,True)
            if _id == Const.DICT_LOST:
                continue
            ids.append(_id)
        return ids

    def fromdct(self,word,flag=True):
        #assert flag == True
        try:
           res=self.dct.token2id[word]
           return res
        except KeyError:
           #continue
           if flag == True:
               pass
           else:
               pass
           res = self.dct.token2id[" "]
           return res

    def gen_eval(self,funcname="gen_eval",columns_name="casdetail",columns_name_tar="",db="myDB",coll="original_data",begin_cursor=0,end_cursor=100, wordFilter=False):
            _ids,_tags,_words = [],[],[]
            _print("\n> gen_train_data new a Eval_Ner()")
            self.get_mongo_coll(db,coll)
            _collections = self.mongo_col
            _count = _collections.count()
            _cursor = _collections.find()
            ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])
            pass # print("\n>len ll after random: ",len(ll))
            while(1):
                for c in ll:
                    i = _cursor[c]
                    _text=self.dwc(i[columns_name])
                    #_crim=self.dwc(i[columns_name_tar])
                    self._vali_type(i,dict,funcname)
                    #print("\n> this is the ",c,'sentence')
                    desdetail_text = self.clr_2_str(_text)
                    if len(re.findall("[\u4e00-\u9fa5]{2,}",desdetail_text))<1:
                        pass # print("\n> text is no here")
                        continue
                    _ids,_tags,_words=[],[],[]
                    for word in list(jieba.cut(desdetail_text)):
                        wd = ""
                        if wordFilter == True:
                            wd = self.fromdct(word)
                            if wd == Const.DICT_LOST:
                                continue
                        else:
                            wd = self.fromdct(word, False)
                        _ids.append(wd)
                        _tags.append(0)
                        _words.append(word)
                    if len(_tags)>lennum:
                        _ids=_ids[:lennum]
                        _tags=_tags[:lennum]
                        _words=_words[:lennum]
                    else:
                        disl=lennum-len(_tags)
                        _ids.extend([self.dct.token2id[" "]]*disl)
                        _tags.extend([0]*disl)
                        _words.extend([" "]*disl)
                    if Const.DEBUG=="True":
                        _print("gen_eval data _ids _tags")
                        _print("_ids,_tags",_ids,_tags)
                        pass#pdb.set_trace()
                    yield _ids,_tags,_words
                    #_ids,_tags,_words = [],[],[]
                #if len(_tags)%(self.btsize*200)==0 and len(_tags)>2:
                #    self.dct.save("./model/my.dct.bak")

    def filter_words(self, filter_words, line):
            #filter_words = Const.VP
            res = []
            words = list(seg.cut(self.dwc(line)))
            for word in words:
                if word.flag in filter_words:
                    res.append(word.word)
            if not len(res)>lennum:
                print("用空格补齐句子")
                res.extend([" "]*1501)
            print(res[:lennum])
            return res[:lennum]

    def line_2_words(self, line):
            res = []
            nr,nz,ns,n,p,a,v,e,nu,x,tt,tt2 = [],[],[],[],[],[],[],[],[],[],[],[]
            words = list(seg.cut(self.dwc(line)))
            print("===")
            print(words)
            #words = words[::-1]
            for word in words:
                #if word.word in self.stw0:
                #    pass#continue
                #if word.word in self.stw3:
                #    pass#continue
                if word.flag in Const.NS:
                    n.append(word.word)
                elif word.flag in Const.NR:
                    n.append(word.word)
                elif word.flag in Const.NZ:
                    n.append(word.word)
                elif word.flag in Const.NOUS:
                    n.append(word.word)
                elif word.flag in Const.PJ:
                    p.append(word.word)
                elif word.flag in Const.AD:
                    a.append(word.word)
                elif word.flag in Const.V:
                    v.append(word.word)
                elif word.flag in Const.ENG:
                    e.append(word.word)
                elif word.flag in Const.NUM:
                    nu.append(word.word)
                elif word.flag in Const.X:
                    x.append(word.word)
                if not word.flag in Const.X:
                    tt.append(word.word)
                    tt2.append(word.flag)
            tt.extend(list("".join(tt)))
            #tt.extend(v)
            #tt.extend(p)
            #tt.extend(n)
            #tt.extend(a)
            #tt.extend(e)
            #tt.extend(nu)
            res.extend(tt)
            if not len(res)>lennum:
                print("用空格补齐句子")
                res.extend([" "]*lennum)
            #pdb.set_trace()
            print(res[:lennum])
            return res[:lennum]

    def text_2_batch(self, textLst):
        resLst = []
        for i in range(64):
            res = []
            if i < len(textLst):
                wordLst = self.line_2_words(textLst[i])
                #wordLst = self.filter_words(Const.VP, textLst[i])
                #wordLst = self.line_2_words(textLst[i])
                for word in wordLst:
                    try:
                        res.append(self.dct.token2id[word])
                    except:
                        res.append(self.dct.token2id[" "])

            else:
                res = [self.dct.token2id[" "] for i in range(lennum)]
            resLst.append(res[:lennum])
        print("text_2_batch return")
        return resLst

    def genDct(self):
        dirpath = "/home/distdev/src/iba/dmp/gongan/labelmarker/data"
        filename = "five_classify_train_250k.txt"
        res , reslb = self.read_file_2_possge(dirpath,filename,1)
        res = [i[-1] for i in res]
        dictionary = gensim.corpora.Dictionary(res)
        dictionary.save('./dct')


    def show_dct(self):
        dirpath = "/home/distdev/src/iba/dmp/gongan/labelmarker/data"
        filename = "five_classify_train_250k.txt"
        res , reslb = self.read_file_2_possge(dirpath,filename,1)
        tfidfHelperInstance = TfidfHelper()
        dictionary = gensim.corpora.dictionary.Dictionary.load("./dictionary.dct2")
        tfidfmodel = gensim.models.tfidfmodel.TfidfModel.load("tfidf.model2")
        print("tfidf 已经保存完毕")
        corpora_words2d = [dictionary.doc2bow(word1d) for word1d in res]
        doc_tfidf_lst = tfidfmodel[corpora_words2d]
        for i in doc_tfidf_lst:
            print(i)
            pdb.set_trace()

    def init_dct(self):
        dirpath = "/home/distdev/src/iba/dmp/gongan/labelmarker/data"
        filename = "five_classify_train_250k.txt"
        print("getCorpus")
        #res , reslb = self.read_file_2_possge(dirpath,filename,1)
        res = []
        lines = open(os.path.join(dirpath,filename)).readlines()
        for line in lines:
            words = list(jieba.cut(line))
            res.append(words)
            res.append(list(line))
        tfidfHelperInstance = TfidfHelper()
        dictionary = gensim.corpora.dictionary.Dictionary(res)
        dictionary.save("./dictionary.dct")
        dictionary = gensim.corpora.dictionary.Dictionary.load("./dictionary.dct")
        tfidfmodel = tfidfHelperInstance.init_tfidf_model(dictionary,res)
        #self.dct = dictionary

    def getCorpus(self, filename, cntround):
        dirpath = "/home/distdev/src/iba/dmp/gongan/labelmarker/data"
        #filename = "five_classify_train_250k.txt"
        print("getCorpus")
        res , reslb = self.read_file_2_possge(dirpath,filename,cntround)

        dictionary = gensim.corpora.dictionary.Dictionary.load("./dictionary.dct")
        self.dct = dictionary
        """
        tfidfHelperInstance = TfidfHelper()
        #dictionary = gensim.corpora.dictionary.Dictionary(res)
        #dictionary.save("./dictionary.dct")

        tfidfmodel = gensim.models.tfidfmodel.TfidfModel.load("tfidf.model")
        #tfidfmodel = tfidfHelperInstance.init_tfidf_model(dictionary,res)
        print("tfidf 已经保存完毕")
        #pdb.set_trace()
        corpora_words2d = [dictionary.doc2bow(word1d) for word1d in res]
        doc_tfidf_lst = tfidfmodel[corpora_words2d]
        result_words = []
        for i in doc_tfidf_lst:
            pick = tfidfHelperInstance.filter_tfidf(i,0.5,dictionary)
            result_words.append(pick)
        res = result_words
        """
        #pdb.set_trace()
        #res = [i[-1] for i in res]
        #linscopy = res.copy()
        #for words2, words in zip(res,linscopy):
        #    for word in words:
        #        #pdb.set_trace()
        #        if word in self.stw0:
        #            words2.remove(word)
        #        elif word in self.stw3:
        #            words2.remove(word)
        #dictionary = self.dct
        #dictionary = self.dct
        # 将文档转换成词袋(bag of words)模型
        print(res)
        docs = []
        for sent in res:
            print(sent,"\n")
            #pdb.set_trace()
            lst = dictionary.doc2idx(sent)
            docs.append(lst)
        result = np.array([])
        resultlst=[]
        result = result.astype(np.int64)
        print(docs)
        print(len(docs))
        yslst = []
        xslst = []
        ys = []
        cnt = 0
        print("ready to while 1")
        for cus in range(len(docs)):
            x_words= docs[cus]
            doc = x_words
            y = reslb[cus]
            ys.append(y)
            xslst.append(x_words)
            #yslst.append(y)
            print("this is the num", y)
            for i in range(lennum):
                word = " "
                if i< len(doc):
                    word = doc[i]
                else:
                    word = doc[i%len(doc)]
                result = np.concatenate([result, np.array([word])])
                if result.shape[0] == 64*lennum:
                    cnt+=1
                    print(">>>>>>>>>batch cnt is ", cnt)
                    resultlst.append(result.reshape(64,lennum))
                    result = np.array([])
                    yslst.append(np.array(ys).reshape([-1]))
                    ys = []
                print(cnt, len(docs))
                if cnt>((len(docs)//64)-1):
                    print('return resultlst yslst')
                    return (resultlst, yslst)
        #return (resultlst, yslst)

    def pickWordFromSent(self, sent, dictionary):
        print(len(sent))
        #pdb.set_trace()
        scores = 0
        for word in sent:
            idn = word[0]
            score = word[1]
            print(idn)
            #pdb.set_trace()
            scores+=score
        print(scores)
        avr = 0.0#0.8*scores/(len(sent)+0.1)
        print(avr)
        filter_ = []
        for word in sent:
            idn = word[0]
            score = word[1]
            if score>avr:
                print(idn)
                print(dictionary[idn])
                w = dictionary[idn]
                if w in self.stw0 or w in self.stw4:
                    continue
                filter_.append(dictionary[idn])
        #pdb.set_trace()
        print(filter_)
        print(sent)
        assert len(filter_)>0
        return filter_

    def tuple2WordsTags(self, tuples, flag=False):
        reswords = []
        restags = []
        dummy = self.fromdct(" ",flag)
        for item in tuples:
            word = item[0]
            tag= item[1]
            wordid = self.fromdct(word,flag)
            if wordid == Const.DICT_LOST:
                continue
            reswords.append(wordid)
            restags.append(self.tags[tag])
        if len(reswords)<lennum:
            l = lennum - len(reswords)
            reswords.extend([dummy]*l)
            restags.extend([0]*l)
        return reswords[:lennum], restags[:lennum]

    def gen_train(self, db='myDB',coll='traindata',textcol='text',targetcol='addrcrim',funcname='gen_train', wordFilter=False, begin_cursor=100, end_cursor=-1):
            _print("\n> gen_train_data new a Eval_Ner()")
            pass#pdb.set_trace()
            self.get_mongo_coll(db,coll)
            _collections=self.mongo_col
            count=_collections.count()
            if end_cursor==-1 or end_cursor>count:
                end_cursor=count
            self._vali_equal(end_cursor,begin_cursor,">",'gen_train')#断言end_cursor>begin_cursor
            #ev = bilstm.eval_bilstm.Eval_Ner()
            #get_mongo_coll( 'myDB', "traindata")
            #_cursor = _collections.find()
            _cursor = _collections.find()
            ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])
            pass # print("\n>len ll after random: ",len(ll))
            for c in ll:
                if c<begin_cursor or c>end_cursor:
                    continue
                #print("\n> this is the num",c,'sentence')
                self._vali_type(c,int,funcname)#断言c是int型数据格式
                #===== 过滤掉字段不全的文本
                item=_cursor[c]
                pass#pdb.set_trace()
                try:
                  if self._vali_in(targetcol,item,'gen_train') == Const.ERROR or \
                      self._vali_in(textcol,item,'gen_train') == Const.ERROR:
                      pass # print(Const.KEYLOSS)
                      continue
                except:
                    pass # print(Const.KEYLOSS)
                    continue
                #===== 过滤掉字数太少的文本 和　无中文 的文本
                pass#pdb.set_trace()
                _crim=self.dwc(item[targetcol])
                _text=self.dwc(item[textcol])
                pass#pdb.set_trace()
                if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1 or len(_text)<3:
                    pass # print("\n> filter text uselessness")
                    pass # print(Const.TEXTUSELESS)
                    continue
                #===== 开始构建 Batch
                _ids, _tags = [], []
                tuple_lst = []
                self._vali_type(_crim,str,funcname)
                _crim_lst=self.toLst(_crim)
                #===== 过滤掉无目标词的文本 将满足要求的地址写入 样本list
                if len(_crim_lst)==0:
                    pass # print(Const.TARGETUSELESS)
                    continue
                _crim_res=[]
                for _ in _crim_lst:
                    if len(_)<3:
                        pass # print(Const.TARGETUSELESS)
                        continue
                    _crim_res.append(self.clr_2_str(_))
                if len(_crim_res)<1:
                    pass # print(Const.TARGETUSELESS)
                    continue
                #=====　文本transform to str and list
                pass#pdb.set_trace()
                txt = self.clr_2_str(_text)
                lsttxt = self.clr_2_lst(_text)
                result = self.split(_crim_res, _text)
                _print(result)
                self._vali_type(result,str,'gen_train')
                tuple_lst = list(re.findall("(.+?)/(.) ", result))
                _print("\n>tuple_lst: ", tuple_lst)

                _ids, _tags = self.tuple2WordsTags(tuple_lst, flag=False)
                if Const.DEBUG=="True":
                    _print("gen_train data _ids _tags")
                    _print("_ids,_tags",_ids,_tags)
                    pass#pdb.set_trace()
                global SAMPLE_CNT
                SAMPLE_CNT.add(c)
                pass#pdb.set_trace()
                _print("\n> there r ", len(SAMPLE_CNT), 'correct sample total here')
                yield _ids,_tags

    def read_file_2_possge(self,dirpath,filename,cntround):
        f = open(os.path.join(dirpath,filename))
        lines = f.readlines()
        np.random.shuffle(lines)
        lth = 0
        if len(lines)>230000:
            lth = 230000
        else:
            lth = len(lines)
        lines = lines[:lth]
        y_lst = []
        x_lst = []
        for i in range(cntround):
            print("sample round", i)
            #randomcnt = np.random.randint(0,len(lines))
            for line in lines:
                x,y = "",""
                try:
                    y=line.split("\t")[0]
                    #if self.LBDCT[y] == "双抢":
                    #    continue
                    x=line.split("\t")[1]

                except:
                    continue
                import pdb
                y = self.crimClassChange(y,self.LBDCT)
                y = re.sub("[^\u4e00-\u9fa5a-zA-Z]","",y)
                y_lst.append(y)
                x_lst.append(x)
        lt = len(y_lst)

        res = []
        reslb = []
        #import pdb
        #pdb.set_trace()
        for i in range(lt):
            print("lt sent",i%1000)
            #cnt = np.random.randint(0, lt)
            cnt = i
            line, lb  = x_lst[cnt], y_lst[cnt]
            #if len(line)<3:
            #    continue
            filted_words = self.line_2_words(line)
            #filted_words = self.filter_words(Const.VP, line)
            res.append(filted_words)
            reslb.append(self.c2n[lb])
        print("read finish")
        return res, reslb


    def read_file_2d_lst(self,dirpath,filename):
        f = open(os.path.join(dirpath,filename))
        #cont = f.read()
        #lines = cont.split("\n")
        lines = f.readlines()
        y_lst = []
        x_lst = []
        for line in lines:
            try:
                y_lst.append(line.split("\t")[0])
                x_lst.append(line.split("\t")[1])
            except:
                continue
        clr_lines = [self.dwc(line) for line in x_lst]
        num_lines = [self.ndwc(line) for line in x_lst]
        char_lines = [self.cdwc(line) for line in x_lst]
        marker_lines = [self.mdwc(line) for line in x_lst]
        #cuts_words = [list(line) for line in clr_lines]
        cuts_words = [list(jieba.cut(line)) for line in clr_lines]

        seg_wds= [list(seg.cut(line)) for line in clr_lines]
        nwds = []
        vwds = []
        pwds = []
        for words in seg_wds:
          nwd = []
          vwd = []
          pwd = []
          for word in words:
            if word.flag in ["n","nz","nr","ns","ad"]:
                #pdb.set_trace()
                nwd.append(word.word)
            elif word.flag in ["vn","v"]:
                vwd.append(word.word)
            elif word.flag in ["a","p"]:
                pwd.append(word.word)
          nwds.append(nwd)
          vwds.append(vwd)
          pwds.append(pwd)
        #cuts_words = []
        #for words__ in cuts_words__:
        #    words = []
        #    for word in words__:
        #        if word in self.stw:
        #            continue
        #        words.append(word)
        #    cuts_words.append(words)
        #self._vali_equal(len(lines), len(cuts_words), "==")
        #pdb.set_trace()
        return nwds, vwds, pwds, cuts_words, num_lines, char_lines, marker_lines, y_lst

    """

    def word_2_vec(self,word):
        try:
            return self.w2vm.get(word)
        except:
            return np.array([-1]*128)

    def words_2_vecs(self,words):
        ids = []
        self._vali_type(words, list)
        while(1):
            for word in words:
                ids.append(self.word_2_vec(word))
            self._vali_equal(len(ids),len(words),"==","words_2_ids")
        return ids
    """

    def words_2_ids(self,seqlen, words, name):
        ids = []
        #self._vali_type(words, list, name)
        #pdb.set_trace()
        for i in range(seqlen):#words:
            if i > len(words)-1:
                ids.append(self.dct.token2id[" "])
            else:
                try:
                    word = words[i]
                    ids.append(self.dct.token2id[word])
                except KeyError:
                    ids.append(self.dct.token2id[" "])
        #self._vali_equal(len(ids),len(words),"==","words_2_ids")
        return ids

    def crimClassChange(self,text,dct):
        if text == "rob":
            print(text)
        return dct[text]

    def get_lb(self):
        filepath = _path("data/lb.txt")
        #filepath = "/home/distdev/iba/dmp/gongan/shandong_crim_classify/data/lb.txt"
        f = open(filepath)
        lines=f.readlines()
        lbs = [re.sub("[^\u4e00-\u9fa5a-zA-Z]","",i) for i in lines]
        c2n={}
        n2c={}
        for i,j in enumerate(lbs):
            c2n[j]=i
            n2c[i]=j
        return c2n,n2c

    def dataGenTrain(self, shuffle=True, begin_cursor=0,dirpath='/home/distdev/src/iba/dmp/gongan/labelmarker/data',filename='five_classify_train_250k.txt',textcol='text',targetcol='addrcrim',funcname='gen_train_text_classify_from_text'):
            funcname = "dataGenTrain"
            res = []
            c2n,n2c=self.get_lb()
            pass # print('this is the func gen_train_text_classify_from_text')
            nwd, vwd, pwd, words2dlst, num_lines, char_lines, marker_lines, y_inputs= self.read_file_2d_lst(dirpath,filename)
            pass # print(words2dlst)
            if Const.DEBUG == True:
               pass # pdb.set_trace()
            count=len(words2dlst)
            end_cursor = count
            #_vali_equal(count,begin_cursor,'>','dataGenTrain')
            if Const.DEBUG == True:
               pass # pdb.set_trace()
            self._vali_equal(count,begin_cursor,">","dataGenTrain")
            self._vali_equal(end_cursor,begin_cursor,">",'dataGenTrain')#断言end_cursor>begin_cursor
            #ev = bilstm.eval_bilstm.Eval_Ner()
            #get_mongo_coll( 'myDB', "traindata")
            #_cursor = _collections.find()
            ll = []
            if shuffle:
                ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])# shuffle list
            else:
                ll = [i for i in range(begin_cursor, end_cursor)]
            pass # print("\n>len ll after random: ",len(ll))
            if Const.DEBUG == True:
               pass # pdb.set_trace()
            #sent_id = np.array([])
            #tags_id = []
            cnt = 0
            dummy = np.array([0.000]*128)
            ll.extend(ll)
            ll.extend(ll)
            for c in ll:
                cnt+=1
                if cnt%100==1:
                    pass#print(cnt)
                print("\n> this is the num",c,'sentence')
                #self._vali_type(c,int,funcname)#断言c是int型数据格式
                #===== 过滤掉字段不全的文本
                sent = words2dlst[c]
                x = ""
                if True:
                    print('sent',sent)
                    x = self.vec(nwd[c], vwd[c], pwd[c], sent, num_lines[c], char_lines[c], marker_lines[c], dummy)
                    #x = self.vec(sent,dummy)
                    print('x',x)
                    x = x.reshape([-1])
                    #sent_id = np.concatenate([sent_id, self.vec(sent,dummy).reshape([-1])])
                else:
                    _words_id = self.words_2_ids(100, sent, funcname)
                    #sent_id.extend(_words_id)
                tag = y_inputs[c]
                _tag_id = c2n[tag]
                y = np.array([_tag_id])
                #tags_id.append(_tag_id)
                #words2idlst.extend(sent)
                if Const.DEBUG == True:
                   pass # pdb.set_trace()
                pass # print("yield x y")
                yield x,y

    def toArr(self,lst,x,y):
        #import pdb
        pass#pdb.set_trace()
        #self._vali_date_lenth(lst,x*y,"toArr()")
        if len(lst)==0:
            return []
        return np.array(lst).reshape(x,y)


    def batch_iter(self,seqlen,gen):
        i=0
        _ids,_tags,_words,_lbs=[],[],[],[]
        while(1):
            _id,_tag,_word,_lb=gen.__next__()
            if Const.DEBUG == True:
               pass # pdb.set_trace()
            self._vali_equal(len(_id), len(_word))
            self._vali_equal(len(_id), seqlen)
            self._vali_equal(len(_tag), self.btsize)
            _ids.extend(_id)
            _tags.extend(_tag)
            _words.extend(_word)
            _lbs.extend(_lb)
            i+=1
            #print("\n>counter:",i)
            if i==self.btsize:
                yield self.toArr(_ids,self.btsize,seqlen), self.toArr(np.one_hot(_tags),self.btsize,18)
                #yield self.toArr(_ids,self.btsize,seqlen), self.toArr(np.one_hot(_tags),self.btsize,18), self.toArr(_words,self.btsize,lennum), self.toArr(_lbs,self.btsize,1)
                _ids,_tags,_words,_lbs=[],[],[],[]
                i=0

    def next_batch_eval(self,seqlen,gen):
        i=0
        _ids,_tags,_words=[],[],[]
        while(1):
            res=gen.__next__()
            if len(res) == 3:
                _id,_tag,_word = res[0],res[1],res[2]
                _ids.extend(_id)
                _tags.extend(_tag)
                _words.extend(_word)
            else:
                _id,_tag = res[0],res[1]
                _ids.extend(_id)
                _tags.extend(_tag)
            i+=1
            pass # print("\n>counter:",i)
            if i==self.btsize:
                yield self.toArr(_ids,self.btsize,seqlen), self.toArr(_tags,self.btsize,seqlen), self.toArr(_words, self.btsize,seqlen)
                if Const.DEBUG=="True":
                    pass # print("next_batch_eval")
                    pass#pdb.set_trace()
                _ids,_tags,_words=[],[],[]
                i=0
            #import pdb
            pass#pdb.set_trace()

    def next_batch(self, _gen):
        round_cnt = 0
        _ids,_tags = [],[]
        while(1):
            _print("next_batch round_cnt", round_cnt)
            try:
                import pdb
                a,b = _gen.__next__()
                _print("\n> a,b the _gen.next() batch")
                _print(a,b)
                #_gen = self.gen_train_data(per=0.8, name=flag)
                #pdb.set_trace()
                assert len(a) == len(b)
                _ids.append(a)
                _tags.append(b)
                assert len(_ids) == len(_tags)
                round_cnt+=1
                if len(_ids)%self.btsize == 0 and len(_ids)>1:
                    if Const.DEBUG=="True":
                        import pdb
                        pass # print("next_batch")
                        pass#pdb.set_trace()
                    pass#pdb.set_trace()
                    yield np.array(_ids).reshape(self.btsize,lennum), np.array(_tags).reshape(self.btsize,lennum)
                    round_cnt=0
                    _ids,_tags = [],[]
            except StopIteration:
                pass#pdb.set_trace()
                traceback.print_exc()
                #round_cnt=0
                _gen = self.gen_train_data("train")
                continue

    #def gen_eval_data(self):
    #    self.gen_train_data(per=0.8, name="eval")
    def merge_coll_mongo(self, lib_nm, col_nm, k1, k2):
        # all items merge into 'addrcrim_sum' k1==>k2
        collection = self.get_mongo_coll(lib_nm, col_nm)
        cursor = self.mongo_col.find()
        for i in  cursor:
                pass # print(i)
                l1 = re.sub("[^\u4e00-\u9fa50-9a-zA-Z,]","",str(i[k1])).split(",")
                l2 = re.sub("[^\u4e00-\u9fa50-9a-zA-Z,]","",str(i[k2])).split(",")
                collection.update_one({"_id":i["_id"]},{"$set":{k1:l1}})
                collection.update_one({"_id":i["_id"]},{"$set":{k2:l2}})
                if len(str(i[k1]))>6:
                    pass#pdb.set_trace()
                    pass # print(">l1: ",l1)
                if len(str(i[k2]))>6:
                    pass#pdb.set_trace()
                    pass # print(">l2: ", l2)
                items = []
                items.extend(l1)
                items.extend(l2)
                if len(items)>0:
                    pass # print("> items: ",items)
                """
                for itema in items:
                    if len(itema)<3:
                        continue
                    for itemb in items:
                         if len(itemb)<3:
                            continue
                         for p in range(1,3):
                             for q in range(1,3):
                                 b = itema.find(itemb[:-q])
                                 if b>-1:
                                     if itema in items_copy:
                                         items_copy.remove(itema)
                                     if itemb in items_copy:
                                         items_copy.remove(itemb)
                                     items_copy.append(itema[b:]+itemb)
                                     break
                """
                pass # print("\n> items", items)
                if "" in items:
                    items.remove("")
                if items == None:
                    items = []
                if len(items)>0:
                   pass # pdb.set_trace()
                items_copy = items.copy()
                items_copy2 = items.copy()
                pass # print("\n> items_copy", items_copy)
                pass # print("\n> items_copy2", items_copy2)
                pass#pdb.set_trace()
                for itema in items_copy2:
                    for itemb in items_copy2:
                        if itema == itemb:
                            continue
                        bina_head = itema.find(itemb)
                        if not bina_head == -1 and (itemb in items_copy):
                            items_copy.remove(itemb)
                items_copy = list(set(items_copy))
                pass # print("\n> items_copy", items_copy)
                if len(items_copy2)>0:# and len(items_copy)==0:
                   pass#pdb.set_trace()
                if len(items_copy)>0:# and len(items_copy)==0:
                   pass#pdb.set_trace()
                collection.update_one({"_id":i["_id"]},{"$set":{k2:items_copy}})

    def replace_kws(self, db, col, key, val1,val2):
        self.get_mongo_coll(db, col).update_many({key:val1},{"$set":{key:val2}})

    def clr_pred(self, db, col, key):
        for i in self.get_mongo_coll(db, col).find():
            ik = self.lst_clr(i[key])
            pass # print(self.get_mongo_coll(db, col).update_one({"_id":i["_id"]},{"$set":{key:ik}}))

    def lst_clr(self, inlst):
        if type(inlst) == str:
            inlst = inlst.split(',')
        kws = ['现金','人民币','公安机关','报案','立案','有限责任公司','手机短信','一条.+?','.+?在.+?时.*?$','.+?服务咨询.*?','.+?一诺财富.+?','.+?有限公司.+?','(?:.+?)(离家出走.+?)','.+?通话时.*?','刑事案件','的一部手机','.+?专用发票.*?','一带.+?','.+?发生冲突','.+?打伤', '卖海洛因', '公司上班期间','上卫生间时','一小区内','笔记本电脑','.*?老板[系是].+?','男子贩毒','的.+?里','被.+?一.+?色.*?手机','民警当场抓获','.{0,4}联系电话','.+?定额.+?','.{0,3}犯罪事实',".{0,3}在网上",'报称其家中','单元防盗门','家[中里]的皮包','.+?面值','.+?税务局定额.*?', '.+?财富.+?','.{0,3}被盗.{0,3}','.{0,3}马路.{0,3}','\d+元','的商品要求依法.+?','^票.+?','.+?余人到我队报称', '[\u4e00-\u9fa5]{0,3}受害人','.+?身份证.+?', '.+?年.+?月.+?日.*?', '.*?短信.*?','一.+?','.*?摩托.*?','最里面','左右','其在','时左右','被人','被男子','有人','.+?[月时].+?时.+?','抓获','离开','贩毒','(?:.+?)(持刀.+?)','^[0-9a-zA-Z]+$']
        outlst = []
        for item in inlst:
            if len(item)<4:
                continue
            for kw in kws:
                item = re.sub(kw,"",item)
            if len(item)>0:
                outlst.append(item)
        return outlst

    def gen_test_data(self):
        """
        generate the test data to feed train model, use memory as small as posible
        """
        pass

    def clr(self, dh):
        dh.clr_pred('myDB', 'traindata', 'addrcrim_sum')

def combine_all():
    dh = Data_Helper()
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred_third","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred_twice","addrcrim_sum")
    dh.merge_coll_mongo('myDB', 'traindata', "addrcrim","addrcrim_sum")
    dh.clr(dh)

def clr_addrcrim_sum():
    dh = Data_Helper()
    dh.clr(dh)
    for i in dh.get_mongo_coll('myDB', 'traindata').find()[22500:21000]:
        pass # print(i['addrcrim_sum'])

def main():
    import pdb
    n=310
    dh=Data_Helper()
    train_gen=dh.gen_train_data("train")
    eval_gen=dh.gen_train_data("eval")
    e = dh.next_batch_eval(eval_gen)
    t = dh.next_batch(train_gen)
    while(n>0):
        a,b = t.__next__()
        c,d,w = e.__next__()
        pass#pdb.set_trace()
        n-=1
    #a,b=dh.next_batch(gen)
    #clr_addrcrim_sum()
    #combine_all()

def arrLowDim(arr, tsne):
    varLowDimArr = tsne.fit_transform(arr)
    #print(varLowDimArr.shape)
    return varLowDimArr

def batches_iter(seqlen, gen, num, tsne):
    embed_size = 128
    nummul = seqlen*num*128
    lenth = 22000//num #len(gen[0])
    #print(lenth//nummul)
    #pdb.set_trace()
    #for i in range(lenth//nummul):
    for i in range(lenth):
        x = np.array([])
        y = np.array([])
        for i in range(num):
            item_0, item_1 = gen.__next__()
            print(item_0)
            pass # print(item_1)
            _x = item_0
            _y = item_1
            y = np.concatenate([y, _y])
            x = np.concatenate([x, _x])
            print(x)
            #print(y)
            #pdb.set_trace()
        #pdb.set_trace()
        x = np.array(x).reshape(num, seqlen, 128)
        y = np.array(y).reshape(num)
        print(x)
        pass # print(y)
        #pdb.set_trace()
        yield (x,y)

if __name__ == "__main__":
    pdb.set_trace()
    sys.argv
    dh=Data_Helper()
    dh.init_dct()
    #dh.show_dct()
    #gen = dh.dataGenTrain(begin_cursor=0,dirpath=_path('data'),filename='five_classify_train_250k.txt',textcol='text',targetcol='addrcrim',funcname='gen_train_text_classify_from_text')

    #pass # pdb.set_trace()
    #res = gen.__next__()

